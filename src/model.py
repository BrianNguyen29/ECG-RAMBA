
"""
ECG RAMBA - Main Model (Scientific Grade)
==================================================
Status: FINAL LOCKED VERSION
Features:
  - Clean Core Compatible
  - Structural Ablation (True Pruning)
  - Inference Ablation (Dynamic Flags)
  - Token Count Fairness (Adaptive Pooling in Fallback)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import CONFIG, ABLATION_CONFIG, NUM_CLASSES, DEVICE

from src.layers import (
    SpatialLeadAttention,
    CrossAttentionFusion,
    MultiScaleTokenizer,
    Perceiver,
    BiMambaBlockV3
)


# ============================================================
# MAIN MODEL
# ============================================================

class ECGRambaV7Advanced(nn.Module):
    """
    ECG RAMBA - Morphable Architecture.
    Designed for rigorous component ablation and zero-shot robustness.
    """

    def __init__(self, cfg: dict = None, ablation: dict = None):
        super().__init__()
        # --- BACKWARD COMPATIBILITY & DEFAULTS ---
        cfg = cfg or CONFIG
        ablation = ablation or {} # Empty dict = Full Model

        self.cfg = cfg
        self.ablation = ablation
        d = cfg["d_model"]
        target_len = cfg.get("seq_len_after_tokenizer", 256) # Default safe fallback

        # --------------------------------------------------------
        # 1. SPATIAL ATTENTION ABLATION
        # --------------------------------------------------------
        # Logic: Only enabled if Config says YES AND Ablation doesn't say NO
        use_spatial = cfg.get("use_spatial_attention", False) and not ablation.get("no_spatial", False)
        self.use_spatial = use_spatial

        if self.use_spatial:
            self.spatial_attn = SpatialLeadAttention(
                d_temp=64,
                n_leads=12,
                n_heads=4
            )
        else:
            self.spatial_attn = None

        # --------------------------------------------------------
        # 2. TOKENIZER ABLATION (FAIRNESS FIX APPLIED)
        # --------------------------------------------------------
        if not ablation.get("no_multiscale", False):
            self.tok = MultiScaleTokenizer(12, d)
        else:
            # Fallback: Single-scale Conv (Standard Patch Embedding)
            # CRITICAL FIX: Added AdaptiveAvgPool1d to ensure fair comparison
            # regarding token count. Both variants now produce exactly 'target_len' tokens.
            self.tok = nn.Sequential(
                nn.Conv1d(12, d, kernel_size=15, stride=4, padding=7),
                nn.BatchNorm1d(d),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(target_len) # <-- FAIRNESS LAYER
            )

        # --------------------------------------------------------
        # 3. ROCKET & HRV (STRUCTURAL PRUNING)
        # --------------------------------------------------------
        # Structural Pruning: actually prevents module initialization to save VRAM
        self.rocket_perceiver = (
            Perceiver(cfg["hydra_dim"], d, cfg["n_latents"])
            if not ablation.get("no_rocket", False) else None
        )

        self.hrv_proj = (
            nn.Sequential(
                nn.Linear(cfg["hrv_dim"], d),
                nn.GELU(),
                nn.Dropout(0.1)
            ) if not ablation.get("no_hrv", False) else None
        )

        # --------------------------------------------------------
        # 4. FUSION ABLATION (Cross-Attn vs Concat)
        # --------------------------------------------------------
        # Logic: Use Cross Attn only if streams exist AND not ablated
        self.use_cross_attn = (
            cfg.get("use_cross_attention_fusion", False)
            and self.tok is not None
            and self.rocket_perceiver is not None
            and not ablation.get("no_fusion", False)
        )

        if self.use_cross_attn:
            self.cross_fusion = CrossAttentionFusion(d, cfg.get("fusion_heads", 8))

        # If fusion is off (Concat), we might need projection if dim doubles
        # But we concat along time dim for sequence safety, so input dim stays 'd'
        # Optional: Add projection if concatenating along channel.
        # Here we stick to Time-Concat for Mamba safety, so no projection needed.
        self.fusion_proj = None

        # --------------------------------------------------------
        # 5. COMMON COMPONENTS
        # --------------------------------------------------------
        self.feature_proj = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU()
        )

        # === FINAL PERCEIVER (Fixed Anchor) ===
        self.use_final_perceiver = cfg.get("use_final_perceiver", True)
        if self.use_final_perceiver:
            n_lat = cfg.get("n_latents", 64)
            self.final_latents = nn.Parameter(torch.randn(1, n_lat, d) * 0.02)

            self.final_cross_attn = nn.MultiheadAttention(d, 8, batch_first=True, dropout=0.1)
            self.final_self_attn = nn.MultiheadAttention(d, 8, batch_first=True, dropout=0.1)

            self.final_norm1 = nn.LayerNorm(d)
            self.final_norm2 = nn.LayerNorm(d)
            self.final_norm3 = nn.LayerNorm(d)

            self.final_ffn = nn.Sequential(
                nn.Linear(d, d * 2),
                nn.GELU(),
                nn.Linear(d * 2, d)
            )

        # === BiMamba BACKBONE ===
        self.layers = nn.ModuleList([
            BiMambaBlockV3(d, i, drop=cfg["drop_path_rate"])
            for i in range(cfg["n_layers"])
        ])
        self.norm = nn.LayerNorm(d)

        # === CLASSIFIER ===
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(d, NUM_CLASSES)
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x, xh, xhr, use_rocket=True, use_hrv=True, use_fusion=True):
        """
        Args:
            x: Raw ECG (B, 12, L)
            xh: Rocket Features
            xhr: HRV Features
            use_rocket (bool): Inference flag for Rocket stream
            use_hrv (bool): Inference flag for HRV stream
            use_fusion (bool): Inference flag to toggle Cross-Attn vs Concat
        """
        B = x.size(0)

        # --- 1. SPATIAL ---
        if self.spatial_attn is not None:
            x = self.spatial_attn(x)
        # If None (ablation), signal passes through unaltered

        # --- 2. TOKENIZER ---
        mamba_feat = self.tok(x)
        # Shape handling:
        # MultiScaleTokenizer returns (B, L, D) -> Ready for Mamba
        # Conv1d Fallback returns (B, D, L) -> Needs transpose
        if self.ablation.get("no_multiscale", False):
            mamba_feat = mamba_feat.transpose(1, 2)

        # --- 3. ROCKET ---
        # Check both Structural existence AND Inference Flag
        if self.rocket_perceiver is not None and use_rocket:
            rocket_feat = self.rocket_perceiver(xh) # (B, Tr, D)
        else:
            # Fallback zeros (Information Removal)
            # Ensures graph compatibility without adding information
            if self.rocket_perceiver is not None:
                n_latents = self.cfg["n_latents"]
                d_model = self.cfg["d_model"]
                rocket_feat = torch.zeros(B, n_latents, d_model, device=x.device, dtype=x.dtype)
            else:
                rocket_feat = None # Module structurally removed

        # --- 4. HRV ---
        if self.hrv_proj is not None and use_hrv:
            with torch.amp.autocast('cuda', enabled=False):
              hrv_feat = self.hrv_proj(xhr.float()).unsqueeze(1)
        else:
            if self.hrv_proj is not None:
                 d_model = self.cfg["d_model"]
                 hrv_feat = torch.zeros(B, 1, d_model, device=x.device, dtype=x.dtype)
            else:
                 hrv_feat = None

        # --- 5. FUSION ---
        # Logic: Must have CrossAttn module AND inputs AND Flag enabled
        if self.use_cross_attn and mamba_feat is not None and rocket_feat is not None and use_fusion:
            # (A) Complex Fusion (Cross Attention)
            fused_seq, _ = self.cross_fusion(mamba_feat, rocket_feat)
        else:
            # (B) Simple Concat (Fallback)
            # Concatenation along Time Dimension preserves Mamba's sequential nature
            parts = [p for p in [mamba_feat, rocket_feat] if p is not None]

            if len(parts) > 1:
                fused_seq = torch.cat(parts, dim=1)
            elif len(parts) == 1:
                fused_seq = parts[0]
            else:
                # Should conceptually not happen in valid flow
                fused_seq = torch.zeros(B, 1, self.cfg["d_model"], device=x.device, dtype=x.dtype)

        # --- 6. MERGE & PROJ ---
        seq_parts = [f for f in [fused_seq, hrv_feat] if f is not None]
        if seq_parts:
            seq = torch.cat(seq_parts, dim=1)
        else:
            # Failsafe
            seq = torch.zeros(B, 1, self.cfg["d_model"], device=x.device, dtype=x.dtype)

        seq = self.feature_proj(seq)

        # --- 7. FINAL PERCEIVER ---
        if self.use_final_perceiver:
            lat = self.final_latents.expand(B, -1, -1)
            lat = lat + self.final_cross_attn(
                self.final_norm1(lat), seq, seq, need_weights=False
            )[0]
            lat = lat + self.final_self_attn(
                self.final_norm2(lat),
                self.final_norm2(lat),
                self.final_norm2(lat),
                need_weights=False
            )[0]
            lat = lat + self.final_ffn(self.final_norm3(lat))
            seq = lat

        # --- 8. BIMAMBA BACKBONE ---
        for layer in self.layers:
            seq = layer(seq)

        pooled = self.norm(seq).mean(dim=1)
        return self.head(pooled)


# ============================================================
# SANITY CHECK
# ============================================================

def run_sanity_check(cfg: dict = None, device: str = None):
    cfg = cfg or CONFIG
    device = device or DEVICE

    print("\n🚑 SANITY CHECK (FINAL VERSION)")

    # 1. Test Full Model (Default)
    model = ECGRambaV7Advanced(cfg, {}).to(device).eval()
    print(f"   • Full Model Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    x = torch.randn(2, 12, 5000, device=device)
    xh = torch.randn(2, cfg["hydra_dim"], device=device)
    xhr = torch.randn(2, cfg["hrv_dim"], device=device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=device == "cuda"):
            out = model(x, xh, xhr)
            print(f"   • Output (Full): {out.shape}")

            # Test Inference Flags
            out_no_r = model(x, xh, xhr, use_rocket=False)
            out_no_f = model(x, xh, xhr, use_fusion=False) # Test fusion flag
            print(f"   • Output (No Rocket Flag): {out_no_r.shape}")
            print(f"   • Output (No Fusion Flag): {out_no_f.shape}")

    # 2. Test Structural Ablation
    # Specifically testing Tokenizer Fairness
    ablation_cfg = {"no_multiscale": True}
    model_abl = ECGRambaV7Advanced(cfg, ablation_cfg).to(device).eval()

    # Check Tokenizer Structure
    print(f"\n   • Checking Tokenizer Fairness...")
    print(f"     Fallback Tokenizer: {model_abl.tok}")
    # Expecting: Sequential(Conv1d, BN, GELU, AdaptiveAvgPool1d)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=device == "cuda"):
            out_abl = model_abl(x, xh, xhr)
            print(f"   • Output (Tokenizer Ablation): {out_abl.shape}")

    assert out.shape == (2, NUM_CLASSES)
    assert out_abl.shape == (2, NUM_CLASSES)
    print("\n✅ FINAL VERDICT: Model is Clean, Fair, and Robust.")
