"""Idempotently add the hypothesis-testing experiment cells to revision notebooks."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"


def source_lines(source: str) -> list[str]:
    lines = source.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    return lines


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source_lines(source)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines(source),
    }


def first_line(cell: dict) -> str:
    return "".join(cell.get("source", [])).splitlines()[0].strip() if cell.get("source") else ""


def upsert_section(path: Path, heading: str, code_source: str, before_heading: str) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    existing = next((index for index, cell in enumerate(cells) if first_line(cell) == heading), None)
    replacement = [markdown(heading), code(code_source)]
    if existing is not None:
        end = existing + 1
        while end < len(cells) and cells[end].get("cell_type") != "markdown":
            end += 1
        cells[existing:end] = replacement
    else:
        insertion = next(
            (index for index, cell in enumerate(cells) if first_line(cell) == before_heading),
            len(cells),
        )
        cells[insertion:insertion] = replacement
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def replace_in_notebook(path: Path, old: str, new: str, *, marker: str | None = None) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    replacements = 0
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if marker is not None and marker not in source:
            continue
        # Check the replacement first because ``new`` may intentionally retain
        # ``old`` as a prefix. This keeps repeated integration runs idempotent.
        if new in source:
            replacements += 1
        elif old in source:
            cell["source"] = source_lines(source.replace(old, new))
            replacements += 1
    if replacements != 1:
        raise RuntimeError(f"Expected one replacement in {path.name}, found {replacements}: {old[:80]!r}")
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


MATCHED_CALIBRATION = r'''# CPU-only secondary OOF-score sensitivity audit. Each evaluated fold excludes its labels from calibrator fitting.
# Base models are not nested-refitted; use the generated claim boundary and do not present this as a deploy-time estimate.
MATCHED_CALIBRATION_N_BOOT = 1000
RUN_MATCHED_CALIBRATION_AUDIT = True

matched_runner = Path('scripts/revision/42_matched_oof_calibration.py')
matched_runner_source = matched_runner.read_text(encoding='utf-8', errors='replace') if matched_runner.is_file() else ''
matched_runner_tokens = [
    'matched_cross_fitted_per_class_monotone_platt_v3',
    'cannot reverse within-fold score ordering',
    'fully nested deploy-time calibration estimate',
    '--reuse-bootstrap',
]
missing_matched_runner_tokens = [token for token in matched_runner_tokens if token not in matched_runner_source]
if missing_matched_runner_tokens:
    raise RuntimeError(
        'Matched calibration runner is stale or missing: '
        + ', '.join(missing_matched_runner_tokens)
    )

matched_required_prediction_paths = {
    'full': 'predictions/oof_final_ema_predictions.npz',
    'minirocket': 'predictions/minirocket_only_oof_predictions.npz',
    'resnet': 'predictions/resnet1d_cnn_oof_predictions.npz',
    'raw_mamba': 'predictions/raw_mamba_oof_predictions.npz',
    'transformer': 'predictions/transformer_ecg_oof_predictions.npz',
}
matched_optional_prediction_paths = {
    'frozen_transform_mlp': 'predictions/hybrid_morphology_oof_predictions.npz',
}
restore_selected_from_mirror(
    list(matched_required_prediction_paths.values())
    + list(matched_optional_prediction_paths.values())
)

def require_canonical_matched_input(relative):
    import hashlib

    manifest_path = stable_mirror / 'manifests' / 'mirror_manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(f'Canonical mirror manifest is missing: {manifest_path}')
    payload = json.loads(manifest_path.read_text(encoding='utf-8'))
    rows = {row.get('relative_path'): row for row in payload.get('artifacts', []) if row.get('relative_path')}
    row = rows.get(relative)
    source = stable_mirror / relative
    active = Path('reports/revision') / relative
    if row is None or not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(f'Matched calibration input is not authenticated by canonical Drive: {relative}')
    def digest(path):
        value = hashlib.sha256()
        with Path(path).open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                value.update(chunk)
        return value.hexdigest()
    expected_sha = str(row.get('sha256') or '')
    expected_size = int(row.get('size_bytes', -1))
    if expected_size != source.stat().st_size or digest(source) != expected_sha:
        raise RuntimeError(f'Canonical mirror contract is invalid for matched input: {relative}')
    if not active.is_file() or active.stat().st_size != expected_size or digest(active) != expected_sha:
        raise RuntimeError(f'Active matched input differs from canonical Drive: {relative}')
    return {'relative_path': relative, 'sha256': expected_sha, 'size_bytes': expected_size}

matched_input_attestations = {}
missing_required_matched_inputs = []
for name, relative in matched_required_prediction_paths.items():
    try:
        matched_input_attestations[name] = require_canonical_matched_input(relative)
    except FileNotFoundError as error:
        missing_required_matched_inputs.append(str(error))
for name, relative in matched_optional_prediction_paths.items():
    try:
        matched_input_attestations[name] = require_canonical_matched_input(relative)
    except FileNotFoundError:
        print(f'Optional matched calibration comparator deferred: {name} ({relative})')
if missing_required_matched_inputs:
    RUN_MATCHED_CALIBRATION_AUDIT = False
    print(
        'Deferred matched calibration audit until Notebook 04 publishes the required OOF baselines:\n - '
        + '\n - '.join(missing_required_matched_inputs)
    )
else:
    print('Canonical matched calibration inputs authenticated:', matched_input_attestations)

matched_model_args = ' '.join(
    f"--model {name}=reports/revision/{contract['relative_path']}"
    for name, contract in matched_input_attestations.items()
)
matched_calibration_command = (
    'python -u scripts/revision/42_matched_oof_calibration.py '
    f'{matched_model_args} --n-boot {MATCHED_CALIBRATION_N_BOOT} '
    '--strict --reuse-bootstrap'
)
print('Matched calibration command:', matched_calibration_command)
if RUN_MATCHED_CALIBRATION_AUDIT:
    run(
        matched_calibration_command,
        log_path='reports/revision/logs/matched_oof_calibration.log',
    )
    run(
        f'python -u scripts/revision/artifact_mirror.py publish --verify-existing size '
        f'--refresh-existing-prefix metrics/matched_calibration_metric_cache '
        f'--refresh-existing-prefix metrics/matched_oof_calibration_ '
        f'--refresh-existing-prefix tables/table_matched_oof_calibration '
        f'--refresh-existing-prefix tables/table_paired_matched_oof_calibration.csv '
        f'--refresh-existing-prefix figures/figure_matched_calibration_audit.png '
        f'--refresh-existing-prefix manifests/matched_oof_calibration_manifest.json '
        f'--refresh-existing-prefix predictions/full_cross_fitted_platt_oof_predictions.npz '
        f'--refresh-existing-prefix predictions/minirocket_cross_fitted_platt_oof_predictions.npz '
        f'--refresh-existing-prefix predictions/resnet_cross_fitted_platt_oof_predictions.npz '
        f'--refresh-existing-prefix predictions/raw_mamba_cross_fitted_platt_oof_predictions.npz '
        f'--refresh-existing-prefix predictions/transformer_cross_fitted_platt_oof_predictions.npz '
        f'--refresh-existing-prefix predictions/frozen_transform_mlp_cross_fitted_platt_oof_predictions.npz '
        f'--mirror-root "{MIRROR_REVISION_ROOT}"',
        log_path='reports/revision/logs/matched_oof_calibration_mirror_publish.log',
    )

matched_outputs = [
    Path('reports/revision/metrics/matched_oof_calibration_summary.json'),
    Path('reports/revision/metrics/matched_oof_calibration_bootstrap.json'),
    Path('reports/revision/tables/table_matched_oof_calibration.csv'),
    Path('reports/revision/tables/table_matched_oof_calibration_coefficients.csv'),
    Path('reports/revision/tables/table_matched_oof_calibration.tex'),
    Path('reports/revision/tables/table_paired_matched_oof_calibration.csv'),
    Path('reports/revision/figures/figure_matched_calibration_audit.png'),
    Path('reports/revision/manifests/matched_oof_calibration_manifest.json'),
]
for path in matched_outputs:
    print(path, 'exists=', path.is_file(), 'size=', path.stat().st_size if path.is_file() else None)
if not RUN_MATCHED_CALIBRATION_AUDIT:
    print(
        'Matched calibration is deferred. Existing files, if any, are not readiness evidence; '
        'Notebook 07 requires the authenticated monotone-Platt v3 protocol.'
    )
if RUN_MATCHED_CALIBRATION_AUDIT and not all(path.is_file() and path.stat().st_size > 0 for path in matched_outputs):
    raise RuntimeError('Matched calibration audit did not produce every required artifact.')
'''


STRUCTURED_ABLATION = r'''# A100 High-RAM. By default, one still-missing fold is trained for the matched Full and every removal variant per run.
# Set ECG_RAMBA_ABLATION_FOLDS_PER_RUN=5 to run all remaining folds continuously.
STRUCTURED_ABLATION_VARIANTS = 'full,no_morphology,no_rhythm,no_context_fusion'
STRUCTURED_ABLATION_FOLDS_PER_RUN = int(os.environ.get('ECG_RAMBA_ABLATION_FOLDS_PER_RUN', '1'))
# This controls only OOF inference/export. Training keeps the canonical batch
# size from configs/config.py so the fresh Full and removals remain matched.
STRUCTURED_ABLATION_OOF_BATCH_SIZE = 128
RUN_STRUCTURED_ABLATION = True

ablation_variants = [item.strip() for item in STRUCTURED_ABLATION_VARIANTS.split(',') if item.strip()]
ablation_checkpoint_root = MIRROR_REVISION_ROOT / 'experimental' / 'structured_ablation_checkpoints'
ablation_fold_cache_root = MIRROR_REVISION_ROOT / 'predictions' / 'structured_ablation_folds'
ablation_pca_cache_root = MIRROR_REVISION_ROOT / 'experimental' / 'structured_ablation_pca_models'
ablation_metric_cache_root = MIRROR_REVISION_ROOT / 'metrics' / 'structured_ablation_metric_cache'
ablation_checkpoint_root.mkdir(parents=True, exist_ok=True)
ablation_fold_cache_root.mkdir(parents=True, exist_ok=True)
ablation_pca_cache_root.mkdir(parents=True, exist_ok=True)
ablation_metric_cache_root.mkdir(parents=True, exist_ok=True)

structured_prediction_relatives = []
for variant in ablation_variants:
    structured_prediction_relatives.extend([
        f'predictions/structured_ablation_{variant}_predictions.npz',
        f'manifests/structured_ablation_{variant}_prediction_run_manifest.json',
    ])
if 'restore_available_revision_artifacts' in globals():
    restore_available_revision_artifacts(structured_prediction_relatives)

ablation_audit_command = (
    'python -u scripts/revision/43_structured_ablation_5fold.py '
    f'--variants {STRUCTURED_ABLATION_VARIANTS} --no-aggregate '
    f'--canonical-model-dir "{model_dir}" --model-root "{ablation_checkpoint_root}" '
    f'--fold-cache-root "{ablation_fold_cache_root}" '
    f'--pca-cache-root "{ablation_pca_cache_root}" '
    f'--metric-cache-dir "{ablation_metric_cache_root}"'
)
run(ablation_audit_command, log_path='reports/revision/logs/structured_ablation_checkpoint_audit.log')
status_table = Path('reports/revision/tables/table_structured_ablation_checkpoint_status.csv')
checkpoint_status = pd.read_csv(status_table)
checkpoint_status['contract_valid'] = checkpoint_status['contract_valid'].astype(str).str.lower().eq('true')
missing_folds = sorted(
    checkpoint_status.loc[~checkpoint_status['contract_valid'], 'fold'].astype(int).unique().tolist()
)
selected_folds = missing_folds[:max(1, STRUCTURED_ABLATION_FOLDS_PER_RUN)]
print('Structured ablation checkpoint root:', ablation_checkpoint_root)
print('Missing fold IDs across removal variants:', missing_folds)
print('Selected folds for this invocation:', selected_folds)

if RUN_STRUCTURED_ABLATION and selected_folds:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Matched five-fold structural retraining requires a GPU runtime; use A100 High-RAM.')
    train_command = (
        'python -u scripts/revision/43_structured_ablation_5fold.py '
        f'--variants {STRUCTURED_ABLATION_VARIANTS} '
        f'--only-folds {",".join(map(str, selected_folds))} --run-training --no-aggregate '
        f'--canonical-model-dir "{model_dir}" --model-root "{ablation_checkpoint_root}" '
        f'--fold-cache-root "{ablation_fold_cache_root}" '
        f'--pca-cache-root "{ablation_pca_cache_root}" '
        f'--metric-cache-dir "{ablation_metric_cache_root}" '
        f'--batch-size {STRUCTURED_ABLATION_OOF_BATCH_SIZE} --num-workers 0'
    )
    run(train_command, log_path='reports/revision/logs/structured_ablation_training.log')

run(ablation_audit_command, log_path='reports/revision/logs/structured_ablation_checkpoint_audit_after_training.log')
checkpoint_status = pd.read_csv(status_table)
checkpoint_status['contract_valid'] = checkpoint_status['contract_valid'].astype(str).str.lower().eq('true')
all_checkpoints_ready = (
    len(checkpoint_status) == len(ablation_variants) * 5
    and bool(checkpoint_status['contract_valid'].all())
)
print('All structured-ablation final EMA checkpoints present:', all_checkpoints_ready)
if RUN_STRUCTURED_ABLATION and all_checkpoints_ready:
    expected_checkpoint_hashes = {
        variant: {
            int(row['fold']): str(row['sha256'])
            for _, row in checkpoint_status.loc[checkpoint_status['variant'].eq(variant)].iterrows()
        }
        for variant in ablation_variants
    }
    structured_prediction_reusable = True
    for variant in ablation_variants:
        prediction_path = Path(f'reports/revision/predictions/structured_ablation_{variant}_predictions.npz')
        run_manifest_path = Path(
            f'reports/revision/manifests/structured_ablation_{variant}_prediction_run_manifest.json'
        )
        if not prediction_path.is_file() or not run_manifest_path.is_file():
            structured_prediction_reusable = False
            break
        run_manifest = json.loads(run_manifest_path.read_text(encoding='utf-8'))
        manifest_hashes = {
            int(row.get('fold', -1)): str(row.get('sha256') or '')
            for row in ((run_manifest.get('inputs') or {}).get('checkpoints') or [])
        }
        prediction_contract = (run_manifest.get('outputs') or {}).get('prediction_file') or {}
        if (
            run_manifest.get('ablation_variant') != variant
            or manifest_hashes != expected_checkpoint_hashes[variant]
            or prediction_contract.get('sha256') != _sha256_local(prediction_path)
        ):
            structured_prediction_reusable = False
            break
    print('Structured-ablation OOF package reusable:', structured_prediction_reusable)
    oof_flag = '' if structured_prediction_reusable else '--run-oof '
    export_command = (
        'python -u scripts/revision/43_structured_ablation_5fold.py '
        f'--variants {STRUCTURED_ABLATION_VARIANTS} {oof_flag}--aggregate --strict-complete '
        f'--canonical-model-dir "{model_dir}" --model-root "{ablation_checkpoint_root}" '
        f'--fold-cache-root "{ablation_fold_cache_root}" '
        f'--pca-cache-root "{ablation_pca_cache_root}" '
        f'--metric-cache-dir "{ablation_metric_cache_root}" '
        f'--batch-size {STRUCTURED_ABLATION_OOF_BATCH_SIZE} --num-workers 0 --n-boot 1000'
    )
    run(export_command, log_path='reports/revision/logs/structured_ablation_oof_and_paired.log')
if RUN_STRUCTURED_ABLATION:
    run(
        f'python -u scripts/revision/artifact_mirror.py publish --verify-existing size '
        f'--refresh-existing-prefix experimental/structured_ablation_checkpoints '
        f'--refresh-existing-prefix experimental/structured_ablation_pca_models '
        f'--refresh-existing-prefix predictions/structured_ablation_folds '
        f'--refresh-existing-prefix metrics/structured_ablation_metric_cache '
        f'--refresh-existing-prefix predictions/structured_ablation_ '
        f'--refresh-existing-prefix metrics/structured_ablation_5fold_summary.json '
        f'--refresh-existing-prefix tables/table_structured_ablation_ '
        f'--refresh-existing-prefix tables/table_paired_structured_ablation_5fold.csv '
        f'--refresh-existing-prefix manifests/structured_ablation_5fold_manifest.json '
        f'--mirror-root "{MIRROR_REVISION_ROOT}"',
        log_path='reports/revision/logs/structured_ablation_mirror_publish.log',
    )

print(status_table, 'exists=', status_table.is_file(), 'size=', status_table.stat().st_size if status_table.is_file() else None)
print('Rerun this cell until missing fold IDs are empty; completed fold checkpoints are reused from canonical Drive.')
'''


PHYSIOLOGICAL_PROBE = r'''# CPU-only once representation embeddings exist. No proxy interval targets are manufactured.
PHYSIOLOGICAL_PROBE_N_BOOT = 1000
RUN_PHYSIOLOGICAL_INTERVAL_PROBE = 'auto'

physiology_relative_outputs = [
    'metrics/physiological_interval_probe_summary.json',
    'tables/table_physiological_interval_probe.csv',
    'tables/table_physiological_interval_target_audit.csv',
    'tables/table_physiological_interval_probe_contrasts.csv',
    'tables/table_physiological_interval_probe.tex',
    'manifests/physiological_interval_probe_manifest.json',
]
restore_notebook06_artifacts(physiology_relative_outputs)

metadata_candidates = [
    Path(os.environ['ECG_RAMBA_PHYSIOLOGY_METADATA']) if os.environ.get('ECG_RAMBA_PHYSIOLOGY_METADATA') else None,
    DRIVE_ROOT / 'physiological_interval_metadata.csv',
    DRIVE_ROOT / 'metadata' / 'physiological_interval_metadata.csv',
]
physiology_metadata = next((path for path in metadata_candidates if path is not None and path.is_file()), None)
physiology_provenance = None
if physiology_metadata is not None:
    print('Physiological metadata candidate:', physiology_metadata)
    print('Physiological metadata SHA256 for provenance sidecar:', sha256_file(physiology_metadata))
    print(
        'The reviewed sidecar must bind this SHA and set independent_of_model_outputs=true and '
        'independent_of_ecg_ramba_feature_cache=true.'
    )
    provenance_candidates = [
        Path(os.environ['ECG_RAMBA_PHYSIOLOGY_PROVENANCE'])
        if os.environ.get('ECG_RAMBA_PHYSIOLOGY_PROVENANCE') else None,
        physiology_metadata.with_suffix(physiology_metadata.suffix + '.provenance.json'),
        physiology_metadata.with_name(physiology_metadata.stem + '_provenance.json'),
    ]
    physiology_provenance = next(
        (path for path in provenance_candidates if path is not None and path.is_file()), None
    )
physiology_command = (
    'python -u scripts/revision/44_physiological_interval_probe.py '
    '--embedding-npz reports/revision/predictions/representation_embeddings_final_ema.npz '
    '--embedding-manifest reports/revision/manifests/representation_embedding_manifest.json '
    f'--n-boot {PHYSIOLOGICAL_PROBE_N_BOOT}'
)
if physiology_metadata is not None:
    physiology_command += f' --metadata-csv "{physiology_metadata}"'
    if physiology_provenance is not None:
        physiology_command += f' --metadata-provenance-json "{physiology_provenance}"'
    else:
        print('Measured metadata found but no reviewed provenance sidecar found; the runner will emit a blocker artifact.')
else:
    print('No reliable measured HR/PR/QRS/QT/QTc metadata CSV found; the runner will emit a blocker artifact.')

physiology_summary_path = Path('reports/revision/metrics/physiological_interval_probe_summary.json')
physiology_manifest_path = Path('reports/revision/manifests/physiological_interval_probe_manifest.json')
physiology_runner_path = Path('scripts/revision/44_physiological_interval_probe.py')
physiology_embedding_path = Path('reports/revision/predictions/representation_embeddings_final_ema.npz')
physiology_embedding_manifest_path = Path('reports/revision/manifests/representation_embedding_manifest.json')
physiology_reusable = False
if physiology_summary_path.is_file() and physiology_manifest_path.is_file():
    existing_summary = json.loads(physiology_summary_path.read_text(encoding='utf-8'))
    existing_manifest = json.loads(physiology_manifest_path.read_text(encoding='utf-8'))
    existing_inputs = existing_manifest.get('inputs') or {}
    existing_outputs = existing_manifest.get('outputs') or {}
    physiology_reusable = (
        existing_summary.get('protocol') == 'fold_held_out_measured_physiological_interval_probe_v3'
        and existing_manifest.get('protocol') == 'fold_held_out_measured_physiological_interval_probe_v3'
        and (existing_manifest.get('runner') or {}).get('sha256') == sha256_file(physiology_runner_path)
        and (existing_inputs.get('embedding') or {}).get('sha256') == sha256_file(physiology_embedding_path)
        and (existing_inputs.get('embedding_manifest') or {}).get('sha256')
        == sha256_file(physiology_embedding_manifest_path)
        and existing_manifest.get('status') == existing_summary.get('status')
        and existing_summary.get('status') in {
            'complete_measured_target_probe',
            'blocked_missing_reliable_interval_metadata',
        }
    )
    physiology_common_output_relatives = [
        relative for relative in physiology_relative_outputs
        if not relative.endswith('.tex') and not relative.startswith('manifests/')
    ]
    physiology_expected_output_relatives = (
        physiology_common_output_relatives + ['tables/table_physiological_interval_probe.tex']
        if existing_summary.get('status') == 'complete_measured_target_probe'
        else physiology_common_output_relatives
    )
    authenticated_output_paths = [
        Path('reports/revision') / relative for relative in physiology_expected_output_relatives
    ]
    physiology_reusable = physiology_reusable and all(
        path.exists()
        and existing_outputs.get(path.as_posix()) == sha256_file(path)
        for path in authenticated_output_paths
    )
    if physiology_metadata is None:
        physiology_reusable = physiology_reusable and existing_summary.get('status') == 'blocked_missing_reliable_interval_metadata'
    elif physiology_provenance is None:
        physiology_reusable = (
            physiology_reusable
            and (existing_inputs.get('metadata') or {}).get('sha256') == sha256_file(physiology_metadata)
            and existing_summary.get('status') == 'blocked_missing_reliable_interval_metadata'
        )
    else:
        physiology_reusable = (
            physiology_reusable
            and (existing_inputs.get('metadata') or {}).get('sha256') == sha256_file(physiology_metadata)
            and (existing_inputs.get('metadata_provenance') or {}).get('sha256') == sha256_file(physiology_provenance)
        )
run_physiology_probe = (
    not physiology_reusable
    if str(RUN_PHYSIOLOGICAL_INTERVAL_PROBE).lower() == 'auto'
    else bool(RUN_PHYSIOLOGICAL_INTERVAL_PROBE)
)
print('Physiological probe reusable:', physiology_reusable, '| should_run=', run_physiology_probe)
print('Physiological probe command:', physiology_command)
if run_physiology_probe:
    run(
        physiology_command,
        log_path='reports/revision/logs/physiological_interval_probe.log',
    )
    run(
        f'python -u scripts/revision/artifact_mirror.py publish --verify-existing size '
        f'--refresh-existing-prefix metrics/physiological_interval_probe_summary.json '
        f'--refresh-existing-prefix tables/table_physiological_interval_ '
        f'--refresh-existing-prefix manifests/physiological_interval_probe_manifest.json '
        f'--mirror-root "{MIRROR_REVISION_ROOT}"',
        log_path='reports/revision/logs/physiological_interval_probe_mirror_publish.log',
    )

if not physiology_summary_path.is_file():
    raise FileNotFoundError(physiology_summary_path)
physiology_summary = json.loads(physiology_summary_path.read_text(encoding='utf-8'))
print(json.dumps({
    'status': physiology_summary.get('status'),
    'targets': physiology_summary.get('targets', []),
    'claim_boundary': physiology_summary.get('claim_boundary'),
}, indent=2))
'''


HYPOTHESIS_LEDGER = r'''# CPU-only synthesis. Strict mode becomes the final gate after Notebook 03/04 and PTB-XL learning-curve rerun.
HYPOTHESIS_LEDGER_STRICT = os.environ.get('ECG_RAMBA_HYPOTHESIS_LEDGER_STRICT', '1') == '1'

hypothesis_command = 'python -u scripts/revision/45_hypothesis_control_claim_boundary.py'
if HYPOTHESIS_LEDGER_STRICT:
    hypothesis_command += ' --strict'
run(
    hypothesis_command,
    log_path='reports/revision/logs/hypothesis_control_claim_boundary.log',
)
hypothesis_table = Path('reports/revision/tables/table_hypothesis_control_finding_claim_boundary.csv')
hypothesis_tex = Path('reports/revision/tables/table_hypothesis_control_finding_claim_boundary.tex')
hypothesis_json = Path('reports/revision/metrics/hypothesis_control_claim_boundary.json')
if not hypothesis_table.is_file() or not hypothesis_tex.is_file() or not hypothesis_json.is_file():
    raise RuntimeError('Hypothesis-control-finding-claim-boundary outputs are missing.')
display(pd.read_csv(hypothesis_table))
print(json.dumps(json.loads(hypothesis_json.read_text(encoding='utf-8')), indent=2)[:8000])

# Refresh the final matrix once so its top-level payload authenticates the ledger
# produced immediately above. The generator keeps incomplete optional experiments
# as claim-specific blockers rather than failing the base rebuttal package.
run(
    'python -u scripts/revision/13_final_evidence_matrix.py --strict',
    log_path='reports/revision/logs/final_evidence_matrix_after_hypothesis_ledger.log',
)

final_evidence_dir = DRIVE_ROOT / 'final_evidence_tables'
final_evidence_dir.mkdir(parents=True, exist_ok=True)
import shutil
presentation_paths = [
    hypothesis_table,
    hypothesis_tex,
    hypothesis_json,
    Path('reports/revision/metrics/matched_oof_calibration_summary.json'),
    Path('reports/revision/metrics/matched_oof_calibration_bootstrap.json'),
    Path('reports/revision/tables/table_matched_oof_calibration.csv'),
    Path('reports/revision/tables/table_matched_oof_calibration_coefficients.csv'),
    Path('reports/revision/tables/table_matched_oof_calibration.tex'),
    Path('reports/revision/tables/table_paired_matched_oof_calibration.csv'),
    Path('reports/revision/figures/figure_matched_calibration_audit.png'),
    Path('reports/revision/metrics/structured_ablation_5fold_summary.json'),
    Path('reports/revision/tables/table_structured_ablation_5fold.csv'),
    Path('reports/revision/tables/table_structured_ablation_5fold.tex'),
    Path('reports/revision/tables/table_paired_structured_ablation_5fold.csv'),
    Path('reports/revision/tables/table_true_fewshot_head_ptbxl_learning_curve.csv'),
    Path('reports/revision/figures/figure_true_fewshot_head_ptbxl_learning_curve.png'),
    Path('reports/revision/metrics/physiological_interval_probe_summary.json'),
    Path('reports/revision/tables/table_physiological_interval_probe.csv'),
    Path('reports/revision/tables/table_physiological_interval_probe_contrasts.csv'),
    Path('reports/revision/tables/table_physiological_interval_probe.tex'),
]
for source in presentation_paths:
    if source.is_file() and source.stat().st_size > 0:
        destination = final_evidence_dir / source.name
        shutil.copy2(source, destination)
        print('Copied:', destination)

run(
    f'python -u scripts/revision/artifact_mirror.py publish --verify-existing size '
    f'--refresh-existing-prefix metrics/hypothesis_control_claim_boundary.json '
    f'--refresh-existing-prefix tables/table_hypothesis_control_finding_claim_boundary. '
    f'--refresh-existing-prefix manifests/hypothesis_control_claim_boundary_manifest.json '
    f'--refresh-existing-prefix metrics/final_evidence_matrix.json '
    f'--refresh-existing-prefix tables/table_final_ '
    f'--refresh-existing-prefix manifests/final_evidence_matrix_manifest.json '
    f'--mirror-root "{MIRROR_REVISION_ROOT}"',
    log_path='reports/revision/logs/hypothesis_control_claim_boundary_mirror_publish.log',
)
'''


def main() -> None:
    replace_in_notebook(
        NOTEBOOK_DIR / "07_results_freeze.ipynb",
        "required_generator_schema = 9",
        "required_generator_schema = 10",
        marker="required_generator_capabilities = {",
    )
    replace_in_notebook(
        NOTEBOOK_DIR / "07_results_freeze.ipynb",
        "    'matched_cross_fitted_calibration',\n",
        "    'matched_cross_fitted_calibration',\n"
        "    'matched_monotone_calibration_v3',\n",
        marker="required_generator_capabilities = {",
    )
    replace_in_notebook(
        NOTEBOOK_DIR / "07_results_freeze.ipynb",
        "    'physiological_interval_probe_gate',\n",
        "    'physiological_interval_probe_gate',\n"
        "    'physiological_interval_probe_v3',\n",
        marker="required_generator_capabilities = {",
    )
    replace_in_notebook(
        NOTEBOOK_DIR / "06_pooling_and_representation.ipynb",
        "'fold_held_out_measured_physiological_interval_probe_v2', 'RUNNER_SOURCE_PATH',\n"
        "        '--embedding-manifest', 'independent_of_model_outputs',",
        "'fold_held_out_measured_physiological_interval_probe_v3', 'RUNNER_SOURCE_PATH',\n"
        "        '--embedding-manifest', 'independent_of_model_outputs',\n"
        "        'independent_of_ecg_ramba_feature_cache', 'metadata_sha256',",
    )
    replace_in_notebook(
        NOTEBOOK_DIR / "02_predictions_and_external_eval.ipynb",
        "    Path('reports/revision/tables/table_true_fewshot_head_ptbxl_primary.csv'),\n"
        "    Path('reports/revision/metrics/true_fewshot_head_ptbxl_bootstrap.json'),",
        "    Path('reports/revision/tables/table_true_fewshot_head_ptbxl_primary.csv'),\n"
        "    Path('reports/revision/tables/table_true_fewshot_head_ptbxl_learning_curve.csv'),\n"
        "    Path('reports/revision/figures/figure_true_fewshot_head_ptbxl_learning_curve.png'),\n"
        "    Path('reports/revision/metrics/true_fewshot_head_ptbxl_bootstrap.json'),",
    )
    replace_in_notebook(
        NOTEBOOK_DIR / "02_predictions_and_external_eval.ipynb",
        "--verify-existing size --mirror-root",
        "--verify-existing size --refresh-existing-prefix predictions/fewshot_head_adaptation_cache "
        "--refresh-existing-prefix metrics/true_fewshot_head_metric_cache --mirror-root",
        marker="true_fewshot_head_ptbxl_mirror_publish.log",
    )
    upsert_section(
        NOTEBOOK_DIR / "03_calibration_and_ci.ipynb",
        "## Matched Cross-Fitted Calibration Audit",
        MATCHED_CALIBRATION,
        "## Summarize Metric Files",
    )
    upsert_section(
        NOTEBOOK_DIR / "04_baselines_and_component_checks.ipynb",
        "## Matched Five-Fold Structured Ablation Runner",
        STRUCTURED_ABLATION,
        "## Fair Baseline Completion Matrix",
    )
    upsert_section(
        NOTEBOOK_DIR / "06_pooling_and_representation.ipynb",
        "## Measured Physiological Interval Probe Gate",
        PHYSIOLOGICAL_PROBE,
        "## Inventory And Stable Mirror",
    )
    upsert_section(
        NOTEBOOK_DIR / "07_results_freeze.ipynb",
        "## Hypothesis-Control-Finding-Claim Boundary Ledger",
        HYPOTHESIS_LEDGER,
        "## Validation Gate",
    )
    print("Updated Notebook 03/04/06/07 hypothesis-testing cells.")


if __name__ == "__main__":
    main()
