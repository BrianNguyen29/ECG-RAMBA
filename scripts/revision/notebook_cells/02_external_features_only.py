# EXTERNAL_FEATURE_PHASE_CELL_V1
# CPU-only phase. Prepare and attest the canonical fixed-seed ROCKET-family,
# fold-PCA, and HRV feature cache; do not construct or evaluate ECG-RAMBA.
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
        "The CPU feature phase requires wfdb. Run Notebook 02 Install Base Dependencies; "
        "Install Model Dependencies is not required."
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

EXTERNAL_FEATURE_PROFILE = os.environ.get(
    "ECG_RAMBA_EXTERNAL_FEATURE_PROFILE", "cpsc_cpu_resume"
).strip().lower()
EXTERNAL_FEATURE_PROFILES = {
    "cpsc_cpu_resume": ["cpsc2021"],
    "ptbxl_georgia_cpu": ["ptbxl", "georgia"],
    "all_external_cpu": ["ptbxl", "georgia", "cpsc2021"],
    "off": [],
}
if EXTERNAL_FEATURE_PROFILE not in EXTERNAL_FEATURE_PROFILES:
    raise ValueError(
        f"Unknown ECG_RAMBA_EXTERNAL_FEATURE_PROFILE={EXTERNAL_FEATURE_PROFILE!r}; "
        f"choose one of {sorted(EXTERNAL_FEATURE_PROFILES)}"
    )
EXTERNAL_FEATURE_DATASETS = EXTERNAL_FEATURE_PROFILES[EXTERNAL_FEATURE_PROFILE]
EXTERNAL_FEATURE_BATCH_SIZE = int(
    os.environ.get("ECG_RAMBA_EXTERNAL_FEATURE_BATCH_SIZE", "64")
)
if not 1 <= EXTERNAL_FEATURE_BATCH_SIZE <= 256:
    raise ValueError("ECG_RAMBA_EXTERNAL_FEATURE_BATCH_SIZE must be in [1, 256]")
print(
    "External CPU feature profile:",
    EXTERNAL_FEATURE_PROFILE,
    "| datasets=",
    EXTERNAL_FEATURE_DATASETS,
    "| batch_size=",
    EXTERNAL_FEATURE_BATCH_SIZE,
)
print(
    "Keep batch_size=64 while the current CPSC hidden .partial checkpoint exists. "
    "Changing it starts a distinct resume contract; 128/256 are only for new caches "
    "after a separate resource check."
)

feature_restore_paths = [
    "predictions/oof_final_ema_predictions.npz",
    "manifests/oof_final_ema_freeze_manifest.json",
    "manifests/oof_final_ema_prediction_run_manifest.json",
    "manifests/fold_pca_manifest.json",
]
feature_restore_paths.extend(
    f"manifests/external_{dataset}_feature_cache_manifest.json"
    for dataset in EXTERNAL_FEATURE_DATASETS
)
feature_restore_args = " ".join(
    f'--include-path "{relative}"' for relative in feature_restore_paths
)
run(
    f"python -u scripts/revision/artifact_mirror.py restore "
    f'--mirror-root "{MIRROR_REVISION_ROOT}" --replace-mismatched '
    f"{feature_restore_args}",
    check=False,
    log_path="reports/revision/logs/external_feature_phase_targeted_restore.log",
)
FOLD_PCA_MANIFEST = Path("reports/revision/manifests/fold_pca_manifest.json")
if EXTERNAL_FEATURE_DATASETS and not (
    FOLD_PCA_MANIFEST.is_file() and FOLD_PCA_MANIFEST.stat().st_size > 0
):
    run(
        "python -u scripts/revision/08_build_fold_pca.py --checkpoint-kind final_ema",
        log_path="reports/revision/logs/external_feature_phase_build_fold_pca.log",
    )
    run(
        "python -u scripts/revision/artifact_mirror.py publish --verify-existing size "
        "--source-conflict-policy source "
        '--include-path "manifests/fold_pca_manifest.json" '
        f'--mirror-root "{MIRROR_REVISION_ROOT}"',
        log_path="reports/revision/logs/external_feature_phase_fold_pca_mirror_publish.log",
    )

from scripts.revision.common import sha256_file

external_feature_runner = Path("scripts/revision/03_generate_external_predictions.py")
current_feature_runner_sha = sha256_file(external_feature_runner)
current_feature_contract = {
    "oof_sha256": sha256_file(
        Path("reports/revision/predictions/oof_final_ema_predictions.npz")
    ),
    "freeze_sha256": sha256_file(
        Path("reports/revision/manifests/oof_final_ema_freeze_manifest.json")
    ),
}


def external_feature_handoff_path(dataset):
    return Path(
        f"reports/revision/manifests/external_{dataset}_feature_cache_manifest.json"
    )


def _is_sha256(value):
    value = str(value or "")
    return len(value) == 64 and all(
        char in "0123456789abcdef" for char in value.lower()
    )


def external_feature_handoff_ready(dataset):
    path = external_feature_handoff_path(dataset)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        cache = payload.get("feature_cache") or {}
        cache_path = Path(str(cache.get("path") or ""))
        archive = payload.get("archive") or {}
        archive_path = Path(str(archive.get("path") or ""))
        records = payload.get("records") or {}
        runtime = payload.get("feature_runtime") or {}
        pca_rows = payload.get("pca") or []
        return bool(
            payload.get("capability") == "external_feature_inference_handoff_v1"
            and payload.get("schema_version") == 1
            and payload.get("status") == "feature_cache_ready"
            and payload.get("dataset") == dataset
            and payload.get("output_tag") == ""
            and payload.get("runner_sha256") == current_feature_runner_sha
            and payload.get("canonical_contract") == current_feature_contract
            and bool(str(payload.get("source_config_hash") or "").strip())
            and cache_path.is_file()
            and cache_path.stat().st_size == int(cache.get("size_bytes", -1))
            and sha256_file(cache_path) == cache.get("sha256")
            and archive_path.is_file()
            and archive_path.stat().st_size == int(archive.get("size_bytes", -1))
            and _is_sha256(archive.get("sha256"))
            and sha256_file(archive_path) == archive.get("sha256")
            and len(pca_rows) == 5
            and all(
                Path(str(row.get("path") or "")).is_file()
                and Path(str(row.get("path") or "")).stat().st_size
                == int(row.get("size_bytes", -1))
                and _is_sha256(row.get("sha256"))
                and sha256_file(Path(str(row.get("path") or "")))
                == row.get("sha256")
                for row in pca_rows
            )
            and int(records.get("count", 0)) > 0
            and _is_sha256(records.get("record_order_sha256"))
            and _is_sha256(records.get("group_order_sha256"))
            and _is_sha256(records.get("split_order_sha256"))
            and bool(records.get("split_ids"))
            and runtime.get("feature_device") == "cpu"
            and runtime.get("backend_contract_capability")
            == "external_rocket_backend_bound_cache_v1"
            and runtime.get("backend_contract_schema_version") == 1
            and bool(str(runtime.get("torch_version") or "").strip())
            and bool(str(runtime.get("numpy_version") or "").strip())
        )
    except Exception as exc:
        print(f"{dataset} feature handoff not reusable:", repr(exc))
        return False


GEORGIA_MAPPING_REVIEW = Path(
    "docs/revision_plan/georgia_label_mapping_review_20260703.csv"
)
GEORGIA_CODE_INVENTORY_OUT = Path(
    "reports/revision/tables/table_georgia_snomed_code_inventory.csv"
)
CPSC_ANNOTATION_AUDIT_OUT = Path(
    "reports/revision/tables/table_cpsc2021_annotation_audit.csv"
)
CPSC_SIGNAL_MEMMAP = (
    MIRROR_REVISION_ROOT
    / "predictions"
    / "cpsc_window_cache"
    / "cpsc2021_preprocessed_windows_source_bound_v3.npy"
)

for dataset in EXTERNAL_FEATURE_DATASETS:
    ready = external_feature_handoff_ready(dataset)
    print(f"{dataset} canonical feature handoff ready={ready}")
    if ready:
        continue
    command = (
        "python -u scripts/revision/03_generate_external_predictions.py "
        f"--dataset {dataset} --features-only --checkpoint-kind final_ema "
        "--feature-device cpu "
        f"--feature-batch-size {EXTERNAL_FEATURE_BATCH_SIZE} "
        "--feature-parity-records 4 --allow-experimental"
    )
    if dataset == "georgia":
        command += (
            f' --georgia-mapping-review "{GEORGIA_MAPPING_REVIEW}"'
            f' --georgia-code-inventory-out "{GEORGIA_CODE_INVENTORY_OUT}"'
        )
    if dataset == "cpsc2021":
        command += (
            f' --cpsc-annotation-audit-out "{CPSC_ANNOTATION_AUDIT_OUT}"'
            f' --cpsc-signal-memmap "{CPSC_SIGNAL_MEMMAP}"'
        )
    run(
        command,
        log_path=f"reports/revision/logs/{dataset}_external_features_only.log",
    )
    handoff = external_feature_handoff_path(dataset)
    publish_args = [
        '--refresh-existing-prefix "predictions/external_feature_cache"',
        f'--include-path "{handoff.relative_to(Path("reports/revision")).as_posix()}"',
    ]
    if dataset == "georgia" and GEORGIA_CODE_INVENTORY_OUT.is_file():
        publish_args.append(
            f'--include-path "{GEORGIA_CODE_INVENTORY_OUT.relative_to(Path("reports/revision")).as_posix()}"'
        )
    if dataset == "cpsc2021":
        publish_args.append(
            "--refresh-existing-prefix "
            '"predictions/cpsc_window_cache/'
            'cpsc2021_preprocessed_windows_source_bound_v3.npy.contract.npz"'
        )
        if CPSC_ANNOTATION_AUDIT_OUT.is_file():
            publish_args.append(
                f'--include-path "{CPSC_ANNOTATION_AUDIT_OUT.relative_to(Path("reports/revision")).as_posix()}"'
            )
    run(
        f"python -u scripts/revision/artifact_mirror.py publish --verify-existing size "
        f'--source-conflict-policy source {" ".join(publish_args)} '
        f'--mirror-root "{MIRROR_REVISION_ROOT}"',
        log_path=(
            f"reports/revision/logs/{dataset}_external_features_only_mirror_publish.log"
        ),
    )
    if not external_feature_handoff_ready(dataset):
        raise RuntimeError(f"{dataset} feature handoff failed post-publish verification")

if EXTERNAL_FEATURE_DATASETS:
    print(
        "CPU FEATURE PHASE COMPLETE: feature caches and handoff manifests are "
        "SHA-verified. Disconnect this runtime and reconnect A100 for "
        "GPU External Prediction Inference."
    )
else:
    print("External CPU feature phase disabled by profile.")
