
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Union

@dataclass
class Mamba2Config:
    d_model: int
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    headdim: int = 64
    ngroups: int = 1
    A_init_range: tuple = (1, 16)
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    conv_bias: bool = True
    bias: bool = False
    use_cuda: bool = False

class Mamba2Simple(nn.Module):
    """
    Pure PyTorch implementation of Mamba2 (State Space Duality).
    Designed to be compatible with mamba_ssm.Mamba2 checkpoints.
    
    FINAL REFINED ARCHITECTURE (Based on Checkpoint Analysis):
    - in_proj: 1676 channels (d_model=384, expand=2, d_state=64, ngroups=1, heads=12)
    - conv1d: 896 channels -> Applies to x, B, C only. (z is skipped)
    - z (768) + xBC (896) + dt (12) = 1676.
    """
    def __init__(
        self,
        d_model,
        d_state=64,
        d_conv=4,
        expand=2,
        headdim=64,
        ngroups=1,
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.headdim = headdim
        self.ngroups = ngroups
        self.d_inner = self.expand * self.d_model
        
        self.nheads = self.d_inner // self.headdim
        assert self.d_inner % self.headdim == 0, "d_inner must be divisible by headdim"

        # Order in Checkpoint: z (d_inner), x (d_inner), B (G*N), C (G*N), dt (H)
        # z: 768
        # x: 768
        # B: 1*64 = 64
        # C: 1*64 = 64
        # dt: 12
        # Total: 1676
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias)

        # Conv applies to x, B, C (896 channels)
        d_conv_in = self.d_inner + 2 * self.ngroups * self.d_state
        
        self.conv1d = nn.Conv1d(
            in_channels=d_conv_in,
            out_channels=d_conv_in,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=d_conv_in,
            padding=d_conv - 1,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.norm = RMSNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

        # Parameters
        self.A_log = nn.Parameter(torch.empty(self.nheads))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.dt_bias = nn.Parameter(torch.empty(self.nheads))
        
        self._init_params(A_init_range, dt_min, dt_max, dt_init_floor)

    def _init_params(self, A_init_range, dt_min, dt_max, dt_init_floor):
        nn.init.uniform_(self.A_log, math.log(A_init_range[0]), math.log(A_init_range[1]))
        dt = torch.exp(
            torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_bias.copy_(inv_dt)

    def forward(self, u):
        batch, seqlen, dim = u.shape

        # 1. Project
        zxbcdt = self.in_proj(u) # (B, L, 1676)
        
        # 2. Split
        # z: first d_inner
        # xBC: next 896
        # dt: last nheads
        
        z = zxbcdt[:, :, :self.d_inner]
        xBC = zxbcdt[:, :, self.d_inner : -self.nheads]
        dt = zxbcdt[:, :, -self.nheads:]
        
        # 3. Conv xBC
        xBC_t = xBC.transpose(1, 2)
        xBC_t = self.conv1d(xBC_t)[:, :, :seqlen]
        xBC = self.act(xBC_t.transpose(1, 2))
        
        # 4. Split x, B, C
        x, B, C = torch.split(xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        
        # 5. SSM
        dt = F.softplus(dt + self.dt_bias)
        x = self.norm(x)
        
        x = x.view(batch, seqlen, self.nheads, self.headdim)
        z = z.view(batch, seqlen, self.nheads, self.headdim)
        
        B = B.view(batch, seqlen, self.ngroups, self.d_state)
        C = C.view(batch, seqlen, self.ngroups, self.d_state)
        
        y = self.ssd_scan(x, dt, self.A_log, B, C, self.D)
        
        # 6. Output Gating
        # z must go through activation? Standard Mamba applies SiLU to z before multiplication.
        # Checkpoint does NOT have z-conv, but does it have z-act?
        # Typically Mamba: z = silu(Linear(u)).
        # Here we did Linear. But no explicit activation on z yet.
        # Apply SiLU to z.
        z = F.silu(z)
        
        y = y * z
        y = y.reshape(batch, seqlen, self.d_inner)
        
        out = self.out_proj(y)
        return out

    def ssd_scan(self, x, dt, A_log, B, C, D):
        """Sequential State Space Duality scan."""
        batch, seqlen, nheads, headdim = x.shape
        _, _, ngroups, d_state = B.shape
        
        heads_per_group = nheads // ngroups
        B_expanded = B.repeat_interleave(heads_per_group, dim=2)
        C_expanded = C.repeat_interleave(heads_per_group, dim=2)
        
        A = -torch.exp(A_log)
        dA = torch.exp(dt * A.view(1, 1, -1))
        
        # Prepare B_dt
        dt_expanded = dt.unsqueeze(-1)
        B_dt = B_expanded * dt_expanded # (B, L, H, N)

        if D is not None:
             D_val = D.view(1, nheads, 1)

        h = torch.zeros(batch, nheads, d_state, headdim, device=x.device)
        ys = []

        for t in range(seqlen):
            decay = dA[:, t, :].unsqueeze(-1).unsqueeze(-1)
            xt = x[:, t, :, :]
            Bt = B_dt[:, t, :, :]
            
            # h[t] = h[t-1] * decay + Bt^T * xt
            update = torch.einsum('bhn,bhp->bhnp', Bt, xt)
            h = h * decay + update
            
            # y[t] = Ct * h[t]
            Ct = C_expanded[:, t, :, :]
            yt = torch.einsum('bhn,bhnp->bhp', Ct, h)
            
            if D is not None:
                 yt = yt + xt * D_val
                 
            ys.append(yt)
        
        y = torch.stack(ys, dim=1)
        return y


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight
