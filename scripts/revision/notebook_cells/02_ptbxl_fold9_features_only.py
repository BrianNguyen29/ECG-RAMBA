# PTBXL_FOLD9_FEATURE_PHASE_CELL_V1
# CPU-only feature preparation for the official PTB-XL fold-9 adaptation pool.
from pathlib import Path
import importlib.util
import json
import os

DRIVE_ROOT = Path(
    globals().get(
        "DRIVE_ROOT",
        Path(os.environ.get("ECG_RAMBA_DRIVE_ROOT", "/content/drive/MyDrive/ECG-Ramba")),
    )
)
REPO_DIR = Path(
    globals().get(
        "REPO_DIR",
        Path(os.environ.get("ECG_RAMBA_REPO_DIR", "/content/ECG-RAMBA")),
    )
)
if not (REPO_DIR / "scripts/revision/03_generate_external_predictions.py").is_file():
    raise FileNotFoundError(
        f"Pinned ECG-RAMBA checkout is unavailable at {REPO_DIR}. Run Notebook 02 Setup first."
    )
if "run" not in globals():
    raise RuntimeError(
        "Run Notebook 02 Setup before this cell so commands receive durable stage/run_id logs."
    )
if importlib.util.find_spec("wfdb") is None:
    raise RuntimeError(
        "Run Notebook 02 Install Base Dependencies. Model Dependencies is not required."
    )
os.chdir(REPO_DIR)

MIRROR_REVISION_ROOT = Path(
    globals().get(
        "MIRROR_REVISION_ROOT",
        DRIVE_ROOT / "revision_artifacts" / "reports" / "revision",
    )
)
EXTERNAL_FEATURE_CACHE_ROOT = (
    MIRROR_REVISION_ROOT / "predictions" / "external_feature_cache"
)
EXTERNAL_FEATURE_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["ECG_RAMBA_EXTERNAL_FEATURE_CACHE_DIR"] = str(EXTERNAL_FEATURE_CACHE_ROOT)

RUN_PTBXL_FOLD9_FEATURES = os.environ.get(
    "ECG_RAMBA_RUN_PTBXL_FOLD9_FEATURES", "1"
).strip().lower() not in {"0", "false", "no", "off"}
PTBXL_FOLD9_FEATURE_BATCH_SIZE = int(
    os.environ.get("ECG_RAMBA_EXTERNAL_FEATURE_BATCH_SIZE", "64")
)
if not 1 <= PTBXL_FOLD9_FEATURE_BATCH_SIZE <= 256:
    raise ValueError("ECG_RAMBA_EXTERNAL_FEATURE_BATCH_SIZE must be in [1, 256]")
PTBXL_FOLD9_FEATURE_HANDOFF = Path(
    "reports/revision/manifests/external_ptbxl_fold9_feature_cache_manifest.json"
)
restore_paths = [
    "predictions/oof_final_ema_predictions.npz",
    "manifests/oof_final_ema_freeze_manifest.json",
    "manifests/oof_final_ema_prediction_run_manifest.json",
    "manifests/fold_pca_manifest.json",
    "manifests/external_ptbxl_fold9_feature_cache_manifest.json",
]
restore_args = " ".join(f'--include-path "{path}"' for path in restore_paths)
run(
    f"python -u scripts/revision/artifact_mirror.py restore "
    f'--mirror-root "{MIRROR_REVISION_ROOT}" --replace-mismatched {restore_args}',
    check=False,
    log_path="reports/revision/logs/ptbxl_fold9_feature_targeted_restore.log",
)
FOLD_PCA_MANIFEST = Path("reports/revision/manifests/fold_pca_manifest.json")
if RUN_PTBXL_FOLD9_FEATURES and not (
    FOLD_PCA_MANIFEST.is_file() and FOLD_PCA_MANIFEST.stat().st_size > 0
):
    run(
        "python -u scripts/revision/08_build_fold_pca.py --checkpoint-kind final_ema",
        log_path="reports/revision/logs/ptbxl_fold9_feature_build_fold_pca.log",
    )
    run(
        "python -u scripts/revision/artifact_mirror.py publish --verify-existing size "
        "--source-conflict-policy source "
        '--include-path "manifests/fold_pca_manifest.json" '
        f'--mirror-root "{MIRROR_REVISION_ROOT}"',
        log_path="reports/revision/logs/ptbxl_fold9_feature_fold_pca_mirror_publish.log",
    )

from scripts.revision.common import sha256_file


def _ptbxl_fold9_is_sha256(value):
    value = str(value or "")
    return len(value) == 64 and all(
        char in "0123456789abcdef" for char in value.lower()
    )


def ptbxl_fold9_feature_handoff_ready():
    try:
        payload = json.loads(PTBXL_FOLD9_FEATURE_HANDOFF.read_text(encoding="utf-8"))
        cache = payload.get("feature_cache") or {}
        cache_path = Path(str(cache.get("path") or ""))
        archive = payload.get("archive") or {}
        archive_path = Path(str(archive.get("path") or ""))
        records = payload.get("records") or {}
        runtime = payload.get("feature_runtime") or {}
        pca_rows = payload.get("pca") or []
        canonical = {
            "oof_sha256": sha256_file(
                Path("reports/revision/predictions/oof_final_ema_predictions.npz")
            ),
            "freeze_sha256": sha256_file(
                Path("reports/revision/manifests/oof_final_ema_freeze_manifest.json")
            ),
        }
        return bool(
            payload.get("capability") == "external_feature_inference_handoff_v1"
            and payload.get("schema_version") == 1
            and payload.get("status") == "feature_cache_ready"
            and payload.get("dataset") == "ptbxl"
            and payload.get("output_tag") == "fold9"
            and payload.get("canonical_contract") == canonical
            and bool(str(payload.get("source_config_hash") or "").strip())
            and payload.get("runner_sha256")
            == sha256_file(Path("scripts/revision/03_generate_external_predictions.py"))
            and cache_path.is_file()
            and cache_path.stat().st_size == int(cache.get("size_bytes", -1))
            and sha256_file(cache_path) == cache.get("sha256")
            and archive_path.is_file()
            and archive_path.stat().st_size == int(archive.get("size_bytes", -1))
            and _ptbxl_fold9_is_sha256(archive.get("sha256"))
            and sha256_file(archive_path) == archive.get("sha256")
            and len(pca_rows) == 5
            and all(
                Path(str(row.get("path") or "")).is_file()
                and Path(str(row.get("path") or "")).stat().st_size
                == int(row.get("size_bytes", -1))
                and _ptbxl_fold9_is_sha256(row.get("sha256"))
                and sha256_file(Path(str(row.get("path") or "")))
                == row.get("sha256")
                for row in pca_rows
            )
            and int(records.get("count", 0)) > 0
            and _ptbxl_fold9_is_sha256(records.get("record_order_sha256"))
            and _ptbxl_fold9_is_sha256(records.get("group_order_sha256"))
            and _ptbxl_fold9_is_sha256(records.get("split_order_sha256"))
            and records.get("split_ids") == ["ptbxl_fold9"]
            and runtime.get("feature_device") == "cpu"
            and runtime.get("backend_contract_capability")
            == "external_rocket_backend_bound_cache_v1"
            and runtime.get("backend_contract_schema_version") == 1
            and bool(str(runtime.get("torch_version") or "").strip())
            and bool(str(runtime.get("numpy_version") or "").strip())
        )
    except Exception as exc:
        print("PTB-XL fold-9 feature handoff not reusable:", repr(exc))
        return False


fold9_feature_ready = ptbxl_fold9_feature_handoff_ready()
print(
    "PTB-XL fold-9 CPU feature handoff ready:",
    fold9_feature_ready,
    "| batch_size=",
    PTBXL_FOLD9_FEATURE_BATCH_SIZE,
)
if RUN_PTBXL_FOLD9_FEATURES and not fold9_feature_ready:
    run(
        "python -u scripts/revision/03_generate_external_predictions.py "
        "--dataset ptbxl --ptbxl-folds 9 --output-tag fold9 --features-only "
        "--checkpoint-kind final_ema --feature-device cpu "
        f"--feature-batch-size {PTBXL_FOLD9_FEATURE_BATCH_SIZE} "
        "--feature-parity-records 4 --allow-experimental",
        log_path="reports/revision/logs/ptbxl_fold9_features_only.log",
    )
    run(
        "python -u scripts/revision/artifact_mirror.py publish --verify-existing size "
        "--source-conflict-policy source "
        '--refresh-existing-prefix "predictions/external_feature_cache" '
        '--include-path "manifests/external_ptbxl_fold9_feature_cache_manifest.json" '
        f'--mirror-root "{MIRROR_REVISION_ROOT}"',
        log_path="reports/revision/logs/ptbxl_fold9_features_only_mirror_publish.log",
    )
    if not ptbxl_fold9_feature_handoff_ready():
        raise RuntimeError(
            "PTB-XL fold-9 feature handoff failed post-publish verification"
        )
    fold9_feature_ready = True
if fold9_feature_ready:
    print(
        "PTB-XL FOLD-9 CPU FEATURE PHASE COMPLETE. Disconnect CPU and reconnect "
        "A100, then run PTB-XL Fold 9 Adaptation-Pool Inference."
    )
elif not RUN_PTBXL_FOLD9_FEATURES:
    print(
        "PTB-XL fold-9 feature phase disabled by "
        "ECG_RAMBA_RUN_PTBXL_FOLD9_FEATURES."
    )
