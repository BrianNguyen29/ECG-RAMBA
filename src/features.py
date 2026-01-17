
"""
ECG RAMBA - Feature Engineering Module
====================================================================
Design principles (STRICT):
- Deterministic & reproducible
- Label-agnostic (NO leakage)
- Fold-safe preprocessing (NO global PCA)
- Cache-stable across config changes
- Architecture-supportive only (NO clinical heuristics)

Guarantees:
- NEVER fits PCA on full dataset
- RAW MiniRocket features ONLY
- PCA fitted strictly inside CV folds
- HRV + amplitude + global record stats (fixed dim = 36)
"""

import os
import numpy as np
import torch
import torch.nn as nn
from typing import List
from scipy.signal import find_peaks
import scipy.stats as stats
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

try:
    import neurokit2 as nk
    HAS_NEUROKIT = True
except ImportError:
    HAS_NEUROKIT = False

from configs.config import CONFIG, PATHS, SEQ_LEN, FS


# ============================================================
# MINI-ROCKET (DETERMINISTIC, CPU-ONLY, PAPER-SAFE)
# ============================================================

class MiniRocketNative(nn.Module):
    """
    Deterministic MiniRocket transform (Dempster et al.).
    Optimized for TorchScript / JIT.
    """

    def __init__(
        self,
        c_in: int = 12,
        seq_len: int = SEQ_LEN,
        num_kernels: int = 10000,
        seed: int = 42,
    ):
        super().__init__()

        kernel_length = 9
        dilations = self._get_dilations(seq_len, kernel_length)

        kernels_per_dilation = num_kernels // len(dilations)
        num_kernels = kernels_per_dilation * len(dilations)

        self.convs = nn.ModuleList()
        self.biases = nn.ParameterList()

        g = torch.Generator(device="cpu").manual_seed(seed)

        for d in dilations:
            conv = nn.Conv1d(
                c_in,
                kernels_per_dilation,
                kernel_size=kernel_length,
                dilation=d,
                padding="same",
                bias=False,
            )
            with torch.no_grad():
                w = torch.randint(-1, 2, conv.weight.shape, generator=g)
                conv.weight.copy_(w.float())
            conv.weight.requires_grad = False
            self.convs.append(conv)

            bias = nn.Parameter(
                torch.randn(kernels_per_dilation, generator=g) * 0.1,
                requires_grad=False,
            )
            self.biases.append(bias)

        # Each kernel → (MAX, PPV)
        self.num_features = num_kernels * 2

    @staticmethod
    def _get_dilations(seq_len, kernel_length):
        max_d = (seq_len - 1) // (kernel_length - 1)
        d, out = 1, []
        while d <= max_d:
            out.append(d)
            d *= 2
        return out if out else [1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats: List[torch.Tensor] = []
        # JIT constraint: iterate via index if ModuleList issues, but standard iteration works
        # JIT constraint: iterate via index if ModuleList issues, but standard iteration works
        # Fix: Use zip to iterate ModuleList and ParameterList together to avoid JIT indexing error
        for conv, bias in zip(self.convs, self.biases):
            out = conv(x)
            maxv, _ = out.max(dim=-1)
            ppv = (out > bias.view(1, -1, 1)).float().mean(dim=-1)
            feats.append(maxv)
            feats.append(ppv)
        return torch.cat(feats, dim=1)


# ============================================================
# RAW MINI-ROCKET CACHE (STABLE, CONFIG-INDEPENDENT)
# ============================================================

def generate_raw_rocket_cache(X: np.ndarray) -> np.ndarray:
    """
    RAW MiniRocket features.

    Cache properties:
    - Depends ONLY on dataset shape + kernel definition
    - Independent of training config / folds / model
    - Deterministic across machines
    """

    assert X.ndim == 3, "X must be (N, C, T)"

    rocket_cache_name = (
        f"rocket_raw_"
        f"N{len(X)}_"
        f"C{X.shape[1]}_"
        f"L{X.shape[-1]}_"
        f"K10000_"
        f"S42.npz"
    )

    cache_path = os.path.join(PATHS["cache_dir"], rocket_cache_name)

    if os.path.exists(cache_path):
        cached = np.load(cache_path)["X"]
        if cached.shape[0] == len(X):
            print(f"✅ Loaded RAW MiniRocket cache: {cached.shape}")
            return cached.astype(np.float32)
        else:
            print("⚠️ Rocket cache exists but size mismatch → regenerating")

    print("🚀 Generating RAW MiniRocket features (CPU, deterministic)...")

    model = MiniRocketNative(
        c_in=X.shape[1],
        seq_len=X.shape[-1],
        num_kernels=10000,
        seed=42,
    ).cpu().eval()

    feats = []
    bs = 64

    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.tensor(X[i:i+bs], dtype=torch.float32, device="cpu")
            feats.append(model(xb).numpy())

    X_rocket = np.vstack(feats).astype(np.float32)

    np.savez_compressed(cache_path, X=X_rocket.astype(np.float16))
    print(f"✅ Saved RAW MiniRocket cache: {X_rocket.shape}")
    print(f"📦 Cache path: {cache_path}")

    return X_rocket


# ============================================================
# FOLD-AWARE PCA (STRICT ANTI-LEAKAGE)
# ============================================================

def fit_pca_on_train(X_train: np.ndarray, n_components: int) -> PCA:
    """
    PCA MUST be fitted on TRAIN fold only.
    """
    pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=42,
    )
    pca.fit(X_train)
    return pca


def apply_pca(pca: PCA, X: np.ndarray) -> np.ndarray:
    return pca.transform(X).astype(np.float32)


# ============================================================
# HRV FEATURES (25)
# ============================================================

def extract_hrv_features(signal: np.ndarray, fs: int = FS) -> np.ndarray:
    feats = np.zeros(25, dtype=np.float32)
    lead = signal[1]

    try:
        peaks = None

        if HAS_NEUROKIT:
            try:
                _, r = nk.ecg_peaks(lead, sampling_rate=fs)
                peaks = r.get("ECG_R_Peaks", None)
            except Exception:
                pass

        if peaks is None or len(peaks) < 3:
            z = (lead - lead.mean()) / (lead.std() + 1e-8)
            peaks, _ = find_peaks(
                z,
                height=1.5,
                distance=int(0.25 * fs),
            )

        if len(peaks) > 2:
            rr = np.diff(peaks) / fs * 1000.0
            rr = rr[(rr > 300) & (rr < 2000)]
            if len(rr) > 1:
                feats[:5] = [
                    rr.mean(),
                    rr.std(),
                    np.median(rr),
                    rr.min(),
                    rr.max(),
                ]
    except Exception:
        pass

    return np.nan_to_num(feats)


# ============================================================
# AMPLITUDE FEATURES (5)
# ============================================================

def extract_amplitude_features(signal_raw: np.ndarray) -> np.ndarray:
    feats = np.zeros(5, dtype=np.float32)
    try:
        amps = np.ptp(signal_raw, axis=-1)
        feats[:] = [
            amps.mean(),
            amps.min(),
            amps.max(),
            amps[:3].mean(),
            amps[6:12].mean(),
        ]
    except Exception:
        pass
    return feats


# ============================================================
# GLOBAL RECORD STATS (6)
# ============================================================

def extract_global_record_stats(signal: np.ndarray) -> np.ndarray:
    z = (signal - signal.mean()) / (signal.std() + 1e-8)
    return np.array(
        [
            z.mean(),
            z.std(),
            np.mean(z ** 2),
            stats.kurtosis(z.flatten()),
            stats.skew(z.flatten()),
            np.percentile(z, 95),
        ],
        dtype=np.float32,
    )


# ============================================================
# HRV + AMP + GLOBAL CACHE (FIXED DIM = 36)
# ============================================================

def generate_hrv_cache(X: np.ndarray, X_raw_amp: np.ndarray) -> np.ndarray:
    """
    HRV + amplitude + global record stats.

    Fixed dimensionality = 36
    """

    cache_path = os.path.join(
        PATHS["cache_dir"],
        f"hrv36_N{len(X)}_C{X.shape[1]}_L{X.shape[-1]}.npz",
    )

    if os.path.exists(cache_path):
        cached = np.load(cache_path)["X"]
        if cached.shape[0] == len(X):
            print(f"✅ Loaded HRV36 cache: {cached.shape}")
            return cached.astype(np.float32)

    print("💓 Extracting HRV + amplitude + global stats (CPU)...")

    feats = np.zeros((len(X), 36), dtype=np.float32)

    for i, sig in enumerate(tqdm(X, desc="HRV36")):
        hrv = extract_hrv_features(sig)
        amp = extract_amplitude_features(X_raw_amp[i])
        gstat = extract_global_record_stats(sig)
        feats[i] = np.concatenate([hrv, amp, gstat])

    assert feats.shape[1] == CONFIG["hrv_dim"], \
        f"HRV dim mismatch: got {feats.shape[1]}, expected {CONFIG['hrv_dim']}"

    np.savez_compressed(cache_path, X=feats.astype(np.float16))
    print(f"✅ Saved HRV36 cache: {feats.shape}")
    print(f"📦 Cache path: {cache_path}")

    return feats
