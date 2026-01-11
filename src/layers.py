
"""
ECG RAMBA - Model Building Blocks
=============================================================
Key guarantees:
- No data leakage
- No clinical heuristics
- Parameter-efficient
- Deterministic & reproducible
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import CONFIG, NUM_CLASSES # NUM_CLASSES imported for downstream head compatibility


# ============================================================
# MAMBA IMPORT
# ============================================================

MAMBA_SOURCE = None

try:
    from mamba_ssm import Mamba2
    MAMBA_CLS = Mamba2
    print("✅ Using Mamba2")
except ImportError:
    try:
        from mamba_ssm import Mamba
        MAMBA_CLS = Mamba
        print("✅ Using Mamba (v1)")
    except ImportError:
        raise ImportError("❌ CRITICAL: mamba_ssm not installed")

BIMAMBA_BWD_OFFSET = 1000 # Offset to ensure unique parameter namespace for backward Mamba


# ============================================================
# LAYER NORM 1D (CHANNEL-WISE)
# ============================================================

class LayerNorm1d(nn.Module):
    """
    Channel-wise normalization for (B, C, L) tensors.
    Note: Normalizes across channel dimension for ECG morphology stability.
    """
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, num_channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


# ============================================================
# CONVNEXT BLOCK
# ============================================================

class ConvNeXtBlock(nn.Module):
    """ConvNeXt-style block for local ECG morphology."""
    def __init__(self, dim, drop=0.):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm1d(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        res = x
        x = self.norm(self.dwconv(x))
        x = x.transpose(1, 2)
        x = self.pwconv2(F.gelu(self.pwconv1(x)))
        return res + self.drop(x.transpose(1, 2))


# ============================================================
# CHANNEL MIX GLU
# ============================================================

class ChannelMixGLU(nn.Module):
    """Channel-wise gated FFN."""
    def __init__(self, d, exp=4):
        super().__init__()
        self.w1 = nn.Linear(d, d * exp)
        self.w2 = nn.Linear(d, d * exp)
        self.w3 = nn.Linear(d * exp, d)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


# ============================================================
# SPATIAL LEAD ATTENTION (FIXED for PTB-XL)
# ============================================================

class SpatialLeadAttention(nn.Module):
    """
    Learned spatial attention across ECG leads.
    Applied strictly in signal space: (B, 12, T)
    """

    def __init__(self, d_temp=64, n_leads=12, n_heads=4):
        super().__init__()

        self.temporal_pool = nn.Sequential(
            nn.Conv1d(1, d_temp // 2, kernel_size=50, stride=25),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Conv1d(d_temp // 2, d_temp, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(4)
        )

        self.lead_embed = nn.Embedding(n_leads, d_temp * 4)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_temp * 4,
            num_heads=n_heads,
            batch_first=True
        )

        self.scale_proj = nn.Linear(d_temp * 4, 1)

    def forward(self, x):
        """
        x: (B, 12, T)
        return: (B, 12, T)
        """
        B, L, T = x.shape
        assert L == self.lead_embed.num_embeddings, \
            f"Expected {self.lead_embed.num_embeddings} leads, got {L}"

        # ---- temporal summarization per lead ----
        # FIX: Use .reshape() instead of .view() to handle non-contiguous memory
        # This is critical for zero-shot inference on PTB-XL
        xt = x.reshape(B * L, 1, T)
        feats = self.temporal_pool(xt).reshape(B, L, -1)  # (B, L, d_temp*4)

        # ---- add learned lead identity ----
        lead_ids = torch.arange(L, device=x.device)
        feats = feats + self.lead_embed(lead_ids)[None, :, :]

        # ---- self-attention across leads ----
        attn_out, _ = self.attn(feats, feats, feats)

        # ---- produce gating scalars per lead ----
        scales = torch.sigmoid(self.scale_proj(attn_out))  # (B, L, 1)

        # ---- multiplicative gating ONLY ----
        return x * scales


# ============================================================
# CROSS-ATTENTION FUSION
# ============================================================

class CrossAttentionFusion(nn.Module):
    """Bidirectional cross-attention between Mamba & Rocket tokens."""
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        self.m2r = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.r2m = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)

        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.n3 = nn.LayerNorm(d_model)
        self.n4 = nn.LayerNorm(d_model)

        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

    def forward(self, mamba, rocket):
        m_enr = mamba + self.m2r(self.n1(mamba), self.n2(rocket), self.n2(rocket), need_weights=False)[0]
        r_enr = rocket + self.r2m(self.n3(rocket), self.n4(mamba), self.n4(mamba), need_weights=False)[0]

        m_pool = m_enr.mean(1)
        r_pool = r_enr.mean(1)
        g = self.gate(torch.cat([m_pool, r_pool], -1))

        fused_pool = g * m_pool + (1 - g) * r_pool
        fused_seq = torch.cat([m_enr, r_enr], dim=1)

        return fused_seq, fused_pool


# ============================================================
# MULTI-SCALE TOKENIZER
# ============================================================

class MultiScaleTokenizer(nn.Module):
    """Multi-scale convolutional tokenizer for ECG."""
    def __init__(self, c_in=12, d_model=384):
        super().__init__()
        d = d_model // 3

        self.fine = nn.Sequential(
            nn.Conv1d(c_in, d, 3, 2, 1), nn.BatchNorm1d(d), nn.GELU(),
            nn.Conv1d(d, d, 3, 2, 1), nn.BatchNorm1d(d), nn.GELU()
        )
        self.medium = nn.Sequential(
            nn.Conv1d(c_in, d, 7, 4, 3), nn.BatchNorm1d(d), nn.GELU(),
            nn.Conv1d(d, d, 5, 2, 2), nn.BatchNorm1d(d), nn.GELU()
        )
        self.coarse = nn.Sequential(
            nn.Conv1d(c_in, d, 15, 8, 7), nn.BatchNorm1d(d), nn.GELU(),
            nn.Conv1d(d, d, 7, 2, 3), nn.BatchNorm1d(d), nn.GELU()
        )

        self.proj = nn.Conv1d(d * 3, d_model, 1)
        self.blocks = nn.Sequential(
            ConvNeXtBlock(d_model, 0.1),
            ConvNeXtBlock(d_model, 0.1)
        )
        self.target_len = CONFIG["seq_len_after_tokenizer"]

    def forward(self, x):
        f = F.adaptive_avg_pool1d(self.fine(x), self.target_len)
        m = F.adaptive_avg_pool1d(self.medium(x), self.target_len)
        c = F.adaptive_avg_pool1d(self.coarse(x), self.target_len)
        x = self.proj(torch.cat([f, m, c], dim=1))
        return self.blocks(x).transpose(1, 2)


# ============================================================
# PERCEIVER
# ============================================================

class Perceiver(nn.Module):
    """
    Parameter-efficient Perceiver IO.
    Latents attend to a SET input (Hydra features).
    """

    def __init__(self, hydra_dim, d_model, n_latents=128):
        super().__init__()
        self.input_proj = nn.Linear(hydra_dim, d_model)
        self.latents = nn.Parameter(torch.randn(1, n_latents, d_model) * 0.02)

        self.cross_attn = nn.MultiheadAttention(d_model, 8, batch_first=True)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(d_model, 8, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x):
        B = x.size(0)
        kv = self.input_proj(x).unsqueeze(1)  # (B, 1, D) set input
        latents = self.latents.expand(B, -1, -1)

        latents = latents + self.cross_attn(
            self.norm_q(latents),
            self.norm_kv(kv),
            self.norm_kv(kv),
            need_weights=False
        )[0]

        latents = latents + self.self_attn(
            self.norm2(latents),
            self.norm2(latents),
            self.norm2(latents),
            need_weights=False
        )[0]

        return latents + self.ffn(self.norm3(latents))


# ============================================================
# BIMAMBA BLOCK
# ============================================================

class BiMambaBlockV3(nn.Module):
    """
    Bidirectional Mamba block with learnable gating.
    Gate is applied BEFORE FFN to stabilize bidirectional fusion.
    """
    def __init__(self, d, layer_idx, drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)

        try:
            self.fwd = MAMBA_CLS(d_model=d, d_state=64, d_conv=4, expand=2, layer_idx=layer_idx)
            self.bwd = MAMBA_CLS(d_model=d, d_state=64, d_conv=4, expand=2,
                                 layer_idx=layer_idx + BIMAMBA_BWD_OFFSET)
        except TypeError:
            self.fwd = MAMBA_CLS(d_model=d, d_state=64, d_conv=4, expand=2)
            self.bwd = MAMBA_CLS(d_model=d, d_state=64, d_conv=4, expand=2)

        self.gate = nn.Sequential(nn.Linear(d * 2, d), nn.Sigmoid())
        self.ffn = ChannelMixGLU(d)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        h = self.norm1(x)
        fwd = self.fwd(h)
        bwd = self.bwd(h.flip(1)).flip(1)
        
        z = self.gate(torch.cat([fwd, bwd], dim=-1))
        x = x + self.drop(z * fwd + (1 - z) * bwd)
        
        return x + self.drop(self.ffn(self.norm2(x)))
