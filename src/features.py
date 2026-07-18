
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
from src.provenance import record_order_fingerprint


HRV36_SCHEMA_VERSION = 2
HRV36_CHECKPOINT_SEMANTICS = "checkpoint_compatible_amplitude_slots_zero"
HRV36_ACTIVE_RR_SLICE = slice(0, 5)
HRV36_RESERVED_SLICE = slice(5, 25)
HRV36_AMPLITUDE_SLICE = slice(25, 30)
HRV36_GLOBAL_SLICE = slice(30, 36)


def checkpoint_compatible_hrv36_contract() -> dict:
    return {
        "schema_version": HRV36_SCHEMA_VERSION,
        "semantics": HRV36_CHECKPOINT_SEMANTICS,
        "active_rr_slots": "0:5",
        "reserved_zero_slots": "5:25",
        "amplitude_zero_slots": "25:30",
        "global_stat_slots": "30:36",
        "full_hrv_claim_supported": False,
    }


def validate_checkpoint_compatible_hrv36(
    features: np.ndarray,
    *,
    context: str = "HRV36",
) -> None:
    """Validate the exact rhythm-feature semantics used by current checkpoints."""
    values = np.asarray(features)
    if values.ndim not in {1, 2} or values.shape[-1] != int(CONFIG["hrv_dim"]):
        raise ValueError(
            f"{context} has shape={values.shape}; expected (..., {CONFIG['hrv_dim']})"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{context} contains non-finite values")
    if np.any(values[..., HRV36_RESERVED_SLICE] != 0):
        raise ValueError(f"{context} reserved slots 5:25 must be zero")
    if np.any(values[..., HRV36_AMPLITUDE_SLICE] != 0):
        raise ValueError(
            f"{context} amplitude slots 25:30 must be zero for the current checkpoint contract"
        )


# ============================================================
# MINI-ROCKET (DETERMINISTIC, CPU-ONLY, PAPER-SAFE)
# ============================================================

class MiniRocketNative(nn.Module):
    """
    Legacy fixed-seed ROCKET-family random convolution transform.

    The class name is retained for checkpoint/cache compatibility.  This is not
    canonical MiniRocket: it uses seeded ternary random kernels, Gaussian
    biases, and MAX+PPV statistics.  Ten thousand requested kernels produce a
    20,000-dimensional output before fold-aware PCA.
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

def generate_raw_rocket_cache(
    X: np.ndarray,
    record_ids: np.ndarray | None = None,
) -> np.ndarray:
    """
    RAW MiniRocket features.

    Cache properties:
    - Depends ONLY on dataset shape + kernel definition
    - Independent of training config / folds / model
    - Deterministic across machines
    """

    assert X.ndim == 3, "X must be (N, C, T)"

    record_fingerprint = (
        record_order_fingerprint(record_ids)
        if record_ids is not None
        else None
    )
    rocket_cache_name = (
        f"rocket_raw_"
        f"N{len(X)}_"
        f"C{X.shape[1]}_"
        f"L{X.shape[-1]}_"
        f"K10000_"
        f"S42"
        f"{f'_R{record_fingerprint}' if record_fingerprint else ''}.npz"
    )

    cache_path = os.path.join(PATHS["cache_dir"], rocket_cache_name)

    if os.path.exists(cache_path):
        with np.load(cache_path, allow_pickle=False) as payload:
            cached = payload["X"]
            cached_record_fingerprint = (
                str(payload["record_order_fingerprint"].item())
                if "record_order_fingerprint" in payload.files
                else None
            )
        expected_shape = (len(X), 20000)
        fingerprint_matches = (
            record_fingerprint is None
            or cached_record_fingerprint == record_fingerprint
        )
        if (
            cached.shape == expected_shape
            and np.isfinite(cached).all()
            and fingerprint_matches
        ):
            print(f"✅ Loaded RAW MiniRocket cache: {cached.shape}")
            return cached.astype(np.float32)
        else:
            print(
                "⚠️ Rocket cache contract mismatch "
                f"(found={cached.shape}, expected={expected_shape}, "
                f"finite={bool(np.isfinite(cached).all())}, "
                f"record_fingerprint_match={fingerprint_matches}) → regenerating"
            )

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
        for i in tqdm(range(0, len(X), bs), desc="MiniRocket", unit="batch"):
            xb = torch.tensor(X[i:i+bs], dtype=torch.float32, device="cpu")
            feats.append(model(xb).numpy())

    X_rocket = np.vstack(feats).astype(np.float32)
    if X_rocket.shape != (len(X), 20000) or not np.isfinite(X_rocket).all():
        raise RuntimeError(
            f"Invalid RAW MiniRocket output: shape={X_rocket.shape}, "
            f"finite={bool(np.isfinite(X_rocket).all())}"
        )

    print(f"💾 Saving RAW MiniRocket cache to: {cache_path}", flush=True)
    cached_float16 = X_rocket.astype(np.float16)
    np.savez_compressed(
        cache_path,
        X=cached_float16,
        storage_dtype=np.asarray("float16"),
        consumer_dtype=np.asarray("float32"),
        quantization_contract=np.asarray("float16_storage_roundtrip_v1"),
        record_order_fingerprint=np.asarray(record_fingerprint or ""),
    )
    print(f"✅ Saved RAW MiniRocket cache: {X_rocket.shape}")
    print(f"📦 Cache path: {cache_path}")

    # Cold-cache and warm-cache runs must consume identical values.
    return cached_float16.astype(np.float32)


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

def generate_hrv_cache(
    X: np.ndarray,
    X_raw_amp: np.ndarray,
    record_ids: np.ndarray | None = None,
    *,
    semantics: str = HRV36_CHECKPOINT_SEMANTICS,
) -> np.ndarray:
    """
    Generate the checkpoint-compatible 36-dimensional rhythm-statistic input.

    Current checkpoints received five RR summaries, twenty reserved zeros, five
    zero amplitude slots, and six global signal statistics. ``X_raw_amp`` is
    retained only for input-length compatibility: the original Chapman pipeline
    passed its precomputed five-dimensional vectors into a waveform amplitude
    extractor, which left slots 25:30 at zero. A corrected/full HRV schema must
    use a new cache namespace and a complete five-fold retraining protocol.
    """

    if semantics != HRV36_CHECKPOINT_SEMANTICS:
        raise ValueError(
            "Unsupported HRV36 semantics for current checkpoints: "
            f"{semantics!r}; expected {HRV36_CHECKPOINT_SEMANTICS!r}"
        )
    if len(X_raw_amp) != len(X):
        raise ValueError(
            f"X_raw_amp length {len(X_raw_amp)} does not match ECG length {len(X)}"
        )

    record_fingerprint = (
        record_order_fingerprint(record_ids)
        if record_ids is not None
        else None
    )
    cache_path = os.path.join(
        PATHS["cache_dir"],
        (
            f"hrv36_N{len(X)}_C{X.shape[1]}_L{X.shape[-1]}"
            f"{f'_R{record_fingerprint}' if record_fingerprint else ''}.npz"
        ),
    )

    if os.path.exists(cache_path):
        with np.load(cache_path, allow_pickle=False) as payload:
            cached = payload["X"]
            cached_record_fingerprint = (
                str(payload["record_order_fingerprint"].item())
                if "record_order_fingerprint" in payload.files
                else None
            )
            cached_semantics = (
                str(payload["hrv_semantics"].item())
                if "hrv_semantics" in payload.files
                else None
            )
        expected_shape = (len(X), int(CONFIG["hrv_dim"]))
        fingerprint_matches = (
            record_fingerprint is None
            or cached_record_fingerprint == record_fingerprint
        )
        semantics_match = cached_semantics in {None, semantics}
        feature_contract_valid = True
        try:
            validate_checkpoint_compatible_hrv36(cached, context=str(cache_path))
        except ValueError:
            feature_contract_valid = False
        if (
            cached.shape == expected_shape
            and np.isfinite(cached).all()
            and fingerprint_matches
            and semantics_match
            and feature_contract_valid
        ):
            provenance = "legacy-value-verified" if cached_semantics is None else "metadata-verified"
            print(
                f"✅ Loaded HRV36 cache: {cached.shape} | semantics={semantics} | {provenance}"
            )
            return cached.astype(np.float32)
        print(
            "⚠️ HRV36 cache contract mismatch "
            f"(found={cached.shape}, expected={expected_shape}, "
            f"finite={bool(np.isfinite(cached).all())}, "
            f"record_fingerprint_match={fingerprint_matches}, "
            f"semantics_match={semantics_match}, "
            f"feature_contract_valid={feature_contract_valid}) → regenerating"
        )

    print(
        "💓 Extracting checkpoint-compatible RR/global statistics "
        "with reserved and amplitude slots fixed to zero (CPU)..."
    )

    feats = np.zeros((len(X), 36), dtype=np.float32)

    for i, sig in enumerate(tqdm(X, desc="HRV36")):
        extracted_hrv = np.asarray(extract_hrv_features(sig), dtype=np.float32)
        if extracted_hrv.shape != (25,):
            raise ValueError(
                f"HRV extractor returned shape={extracted_hrv.shape}; expected (25,)"
            )
        hrv = np.zeros(25, dtype=np.float32)
        hrv[:5] = extracted_hrv[:5]
        amp = np.zeros(5, dtype=np.float32)
        gstat = extract_global_record_stats(sig)
        feats[i] = np.concatenate([hrv, amp, gstat])

    assert feats.shape[1] == CONFIG["hrv_dim"], \
        f"HRV dim mismatch: got {feats.shape[1]}, expected {CONFIG['hrv_dim']}"
    if not np.isfinite(feats).all():
        raise RuntimeError("HRV36 extraction produced non-finite values")
    validate_checkpoint_compatible_hrv36(feats, context="generated HRV36")

    print(f"💾 Saving HRV36 cache to: {cache_path}", flush=True)
    cached_float16 = feats.astype(np.float16)
    np.savez_compressed(
        cache_path,
        X=cached_float16,
        storage_dtype=np.asarray("float16"),
        consumer_dtype=np.asarray("float32"),
        quantization_contract=np.asarray("float16_storage_roundtrip_v1"),
        record_order_fingerprint=np.asarray(record_fingerprint or ""),
        hrv_semantics=np.asarray(semantics),
        hrv_schema_version=np.asarray(HRV36_SCHEMA_VERSION, dtype=np.int16),
        active_rr_slots=np.asarray("0:5"),
        reserved_zero_slots=np.asarray("5:25"),
        amplitude_zero_slots=np.asarray("25:30"),
        global_stat_slots=np.asarray("30:36"),
    )
    print(f"✅ Saved HRV36 cache: {feats.shape}")
    print(f"📦 Cache path: {cache_path}")

    return cached_float16.astype(np.float32)
