"""Idempotently integrate forensic runtime/storage gates into notebooks 00--07."""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
MAMBA_MARKER = "MAMBA_INSTALLER_CAPABILITY = 'ecg_ramba_mamba_installer_v1'"
MAMBA_SCHEMA_MARKER = "MAMBA_INSTALLER_SCHEMA_VERSION = 1"
BASE_INSTALLER_MARKER = "BASE_INSTALLER_CAPABILITY = 'ecg_ramba_base_installer_v1'"
BASE_INSTALLER_SCHEMA_MARKER = "BASE_INSTALLER_SCHEMA_VERSION = 1"
RUN_HISTORY_MARKER = "FORENSIC_RUN_HISTORY_CAPABILITY = 'stage_run_id_v1'"
AUTHORITY_MARKER = "FORENSIC_CODE_AUTHORITY_CAPABILITY = 'canonical_git_commit_pin_v1'"
AUTHORITY_SCHEMA_MARKER = "FORENSIC_CODE_AUTHORITY_SCHEMA_VERSION = 1"
AUTHORITY_BLOCK_START = "# BEGIN FORENSIC CODE AUTHORITY PIN"
AUTHORITY_BLOCK_END = "# END FORENSIC CODE AUTHORITY PIN"
AUTHENTICATED_BOOTSTRAP_UNIT = "authenticated_source_patient_record"
GROUP_SEMANTICS = "physionet_ecg_arrhythmia_one_patient_per_record_v1"
GROUP_REFERENCE = "https://physionet.org/content/ecg-arrhythmia/1.0.0/"
NOTEBOOK02_PREFLIGHT_START = "# BEGIN FORENSIC NOTEBOOK 02 CAPABILITY PREFLIGHT"
NOTEBOOK02_PREFLIGHT_END = "# END FORENSIC NOTEBOOK 02 CAPABILITY PREFLIGHT"


NOTEBOOK02_CAPABILITY_PREFLIGHT = r'''# BEGIN FORENSIC NOTEBOOK 02 CAPABILITY PREFLIGHT
# The immutable authority checkout is the source of code. Validate reviewed,
# machine-readable capabilities without downloading mutable branch files into
# the detached checkout.
import ast as _compat_ast

REVISION_CAPABILITY_REQUIREMENTS = {
    'scripts/revision/03_generate_external_predictions.py': {
        'NOTEBOOK_02_EXTERNAL_EXPORT_CAPABILITY': 'external_export_full10s_grouped_v1',
        'NOTEBOOK_02_EXTERNAL_EXPORT_SCHEMA_VERSION': 1,
    },
    'scripts/revision/18_external_protocol_gate.py': {
        'NOTEBOOK_02_EXTERNAL_GATE_CAPABILITY': 'external_gate_full10s_grouped_v1',
        'NOTEBOOK_02_EXTERNAL_GATE_SCHEMA_VERSION': 1,
    },
    'scripts/revision/external_reuse_contract.py': {
        'EXTERNAL_REUSE_CAPABILITY': 'source_bound_external_reuse_v1',
        'EXTERNAL_REUSE_SCHEMA_VERSION': 1,
    },
    'scripts/revision/49_build_oof_group_sidecar.py': {
        'GROUP_SIDECAR_CAPABILITY': 'chapman_oof_group_sidecar_v1',
        'GROUP_SIDECAR_SCHEMA_VERSION': 1,
    },
}
REVISION_TOKEN_REQUIREMENTS = {
    'scripts/revision/common.py': [
        'one-label mapped task retains positive-label',
        'average="binary"',
    ],
    'scripts/revision/artifact_mirror.py': [
        '--verify-existing',
        '--include-prefix',
        'discovered_unmanifested_count',
        'recoverable_publish_transaction_v1',
        'PUBLISH_TRANSACTION_NAME',
    ],
    'scripts/revision/01_generate_predictions.py': [
        '--fold-cache-dir',
        'OOF_CACHE_PROVENANCE_SCHEMA_VERSION',
        'cache_contract_sha256',
    ],
    'scripts/revision/03_generate_external_predictions.py': [
        '--georgia-mapping-review',
        '--georgia-code-inventory-out',
        '--cpsc-annotation-audit-out',
        'validate_checkpoint_files_against_oof_run_manifest',
    ],
    'scripts/revision/06_freeze_oof.py': [
        '--check-existing-freeze',
        'Validated without rewrite',
        '--metadata-refresh-from-existing-oof',
        'verified_metadata_only_refresh',
        '--manuscript-ready-strict',
        'source_archive_sha256',
    ],
    'scripts/revision/18_external_protocol_gate.py': [
        'georgia_mapping_inventory',
        'cpsc_annotation_audit',
        'metric_implementation_sha256',
        'aggregation_implementation_sha256',
        'positive_label_multilabel_reduction',
    ],
    'scripts/revision/31_generate_external_comparator_predictions.py': [
        'validate_checkpoint_set',
        '--fold-cache-dir',
        'Verified final external comparator artifacts were not reusable',
    ],
    'scripts/revision/32_paired_external_comparators.py': [
        'Paired external bootstrap requires at least two groups',
        'probabilities must be in [0,1]',
    ],
    'scripts/revision/33_group_safe_score_calibration.py': [
        '--primary-fraction',
        'pre_specified_before_test_metric_evaluation',
        'independent_target_groups_from_adaptation_pool',
    ],
    'scripts/revision/34_extract_external_representations.py': [
        'checkpoint_source_contract',
        'validate_checkpoint_files_against_oof_run_manifest',
        'frozen_encoder_external_record_representation_v2_source_bound',
        'validate_source_provenance',
        'current runner/canonical contract',
    ],
    'scripts/revision/35_true_fewshot_head_adaptation.py': [
        'embedding_manifest_path',
        'Embedding manifest is stale or incomplete',
        'independent_target_groups_from_adaptation_pool',
        'REPRESENTATION_PROTOCOL_VERSION = 2',
        'pre_specified_before_test_metric_evaluation',
        'primary_endpoint_rows',
    ],
}
REVISION_REQUIRED_FILES = [
    'docs/revision_plan/georgia_label_mapping_review_20260703.csv',
]


def _literal_module_assignments(path):
    try:
        tree = _compat_ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    except Exception as exc:
        raise RuntimeError(f'Could not parse capability source {path}: {exc}') from exc
    assignments = {}
    for node in tree.body:
        if isinstance(node, _compat_ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], _compat_ast.Name):
            try:
                assignments[node.targets[0].id] = _compat_ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
        elif isinstance(node, _compat_ast.AnnAssign) and isinstance(node.target, _compat_ast.Name):
            try:
                assignments[node.target.id] = _compat_ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
    return assignments


compatibility_failures = []
for rel_path, expected in REVISION_CAPABILITY_REQUIREMENTS.items():
    path = REPO_DIR / rel_path
    if not path.is_file() or path.stat().st_size == 0:
        compatibility_failures.append(f'{rel_path}: missing or empty')
        continue
    observed = _literal_module_assignments(path)
    mismatches = {
        key: {'expected': value, 'observed': observed.get(key)}
        for key, value in expected.items()
        if observed.get(key) != value
    }
    if mismatches:
        compatibility_failures.append(f'{rel_path}: capability mismatch {mismatches}')

for rel_path, required_tokens in REVISION_TOKEN_REQUIREMENTS.items():
    path = REPO_DIR / rel_path
    if not path.is_file() or path.stat().st_size == 0:
        compatibility_failures.append(f'{rel_path}: missing or empty')
        continue
    source_text = path.read_text(encoding='utf-8', errors='replace')
    missing_tokens = [token for token in required_tokens if token not in source_text]
    if missing_tokens:
        compatibility_failures.append(f'{rel_path}: missing contract tokens {missing_tokens}')

for rel_path in REVISION_REQUIRED_FILES:
    path = REPO_DIR / rel_path
    if not path.is_file() or path.stat().st_size == 0:
        compatibility_failures.append(f'{rel_path}: missing or empty')

if compatibility_failures:
    authority = (CODE_AUTHORITY or {}).get('commit', 'unknown')
    raise RuntimeError(
        f'Notebook 02 is incompatible with pinned authority {authority}: '
        + ' ; '.join(compatibility_failures)
        + '. Do not hotfix a detached authority checkout from mutable GitHub main. '
        'Commit and review the required change, then rotate authority through Notebook 00.'
    )
print('Revision capability preflight: OK')
# END FORENSIC NOTEBOOK 02 CAPABILITY PREFLIGHT
'''


NOTEBOOK02_EXTERNAL_REUSE_FUNCTION = r'''def external_prediction_ready(dataset):
    from scripts.revision.external_reuse_contract import validate_external_prediction_reuse

    mirror_root = Path(globals().get(
        'MIRROR_REVISION_ROOT',
        DRIVE_ROOT / 'revision_artifacts' / 'reports' / 'revision',
    ))
    result = validate_external_prediction_reuse(
        dataset,
        revision_root=Path('reports/revision'),
        archive_path=external_archives.get(dataset),
        exporter_path=Path('scripts/revision/03_generate_external_predictions.py'),
        oof_path=Path('reports/revision/predictions/oof_final_ema_predictions.npz'),
        freeze_path=Path('reports/revision/manifests/oof_final_ema_freeze_manifest.json'),
        archive_hash_cache_dir=mirror_root / 'manifests' / 'external_archive_hash_cache',
        threshold=0.5,
        q=3.0,
    )
    reasons = list(result.get('reasons', []))
    external_reuse_diagnostics[dataset] = reasons
    print(f'{dataset} external source-bound reuse contract ready={result.get("ready", False)}')
    for path in external_required_artifacts(dataset):
        print(f'  {path}: exists={path.exists()} size={path.stat().st_size if path.exists() else None}')
    if reasons:
        print('  reuse rejected:', '; '.join(reasons))
    diagnostics = result.get('diagnostics', {})
    if diagnostics.get('q3_reconstruction_max_abs') is not None:
        print('  Q=3 reconstruction max_abs:', diagnostics['q3_reconstruction_max_abs'])
    return bool(result.get('ready', False))
'''


NOTEBOOK02_OOF_SIDECAR_PREFLIGHT = r'''
# CPU-only OOF provenance restore. A missing group sidecar is a metadata-contract
# gap, not evidence that model inference must be repeated.
CANONICAL_GROUP_SIDECAR = Path('reports/revision/manifests/oof_final_ema_group_sidecar.npz')
OOF_CORE_ARTIFACTS = [
    Path(f'reports/revision/predictions/{CANONICAL_OOF_STEM}_predictions.npz'),
    Path(f'reports/revision/predictions/{CANONICAL_OOF_STEM}_slice_predictions.npz'),
    Path(f'reports/revision/metrics/{CANONICAL_OOF_STEM}_prediction_summary.json'),
    Path(f'reports/revision/tables/{CANONICAL_OOF_STEM}_class_summary.csv'),
    Path(f'reports/revision/manifests/{CANONICAL_OOF_STEM}_prediction_run_manifest.json'),
]
OOF_CONTRACT_RESTORE_PATHS = [
    *OOF_CORE_ARTIFACTS,
    CANONICAL_FREEZE_MANIFEST,
    CANONICAL_GROUP_SIDECAR,
    Path('reports/revision/logs/oof_final_ema_generate_predictions.log'),
]
oof_restore_args = ' '.join(
    f'--include-path "{path.relative_to(Path("reports/revision")).as_posix()}"'
    for path in OOF_CONTRACT_RESTORE_PATHS
)
run(
    f'python -u scripts/revision/artifact_mirror.py restore --mirror-root "{stable_mirror}" '
    f'--replace-mismatched {oof_restore_args}',
    log_path='reports/revision/logs/oof_contract_targeted_restore.log',
)


def ensure_oof_group_sidecar():
    record_path = OOF_CORE_ARTIFACTS[0]
    if not record_path.is_file() or record_path.stat().st_size == 0:
        print('OOF group sidecar build deferred because canonical OOF predictions are absent.')
        return False
    run(
        'python -u scripts/revision/49_build_oof_group_sidecar.py '
        f'--oof-predictions "{record_path}" --source-archive "{chapman_zip}" '
        f'--out "{CANONICAL_GROUP_SIDECAR}" --expected-records 44186 --reuse-existing',
        log_path='reports/revision/logs/oof_group_sidecar.log',
    )
    run(
        f'python -u scripts/revision/artifact_mirror.py publish --verify-existing size '
        f'--source-conflict-policy source '
        f'--include-path "manifests/oof_final_ema_group_sidecar.npz" '
        f'--mirror-root "{stable_mirror}"',
        log_path='reports/revision/logs/oof_group_sidecar_immediate_mirror_publish.log',
    )
    return True


ensure_oof_group_sidecar()
'''


RUN_HISTORY_WRAPPER = r'''

# Forensic run-history wrapper. The legacy helper writes live output while this
# wrapper gives every invocation a unique, durable stage/run_id log and retains
# the requested stable path as the latest-run convenience copy.
FORENSIC_RUN_HISTORY_CAPABILITY = 'stage_run_id_v1'
_forensic_base_run = run

def run(cmd, check=True, log_path=None, tail_lines=160, cwd=None):
    import os as _forensic_os
    import shutil as _forensic_shutil
    import subprocess as _forensic_subprocess
    import time as _forensic_time
    import uuid as _forensic_uuid
    from datetime import datetime as _forensic_datetime, timezone as _forensic_timezone
    from pathlib import Path as _ForensicPath

    run_cwd = _ForensicPath(cwd) if cwd else _ForensicPath.cwd()
    if log_path is None:
        stable_log = run_cwd / 'reports' / 'revision' / 'logs' / 'notebook_command_latest.log'
    else:
        stable_log = _ForensicPath(log_path)
        if not stable_log.is_absolute():
            stable_log = run_cwd / stable_log
    stable_log.parent.mkdir(parents=True, exist_ok=True)
    stage = stable_log.stem
    run_id = _forensic_datetime.now(_forensic_timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '-' + _forensic_uuid.uuid4().hex[:10]
    history_log = stable_log.parent / 'history' / stage / f'{run_id}.log'
    history_log.parent.mkdir(parents=True, exist_ok=True)

    canonical_root = globals().get('MIRROR_REVISION_ROOT')
    if canonical_root is None and 'DRIVE_ROOT' in globals():
        canonical_root = _ForensicPath(DRIVE_ROOT) / 'revision_artifacts' / 'reports' / 'revision'
    canonical_history = None
    if canonical_root is not None:
        canonical_root = _ForensicPath(canonical_root)
        canonical_history = canonical_root / 'logs' / 'history' / stage / f'{run_id}.log'
        canonical_history.parent.mkdir(parents=True, exist_ok=True)

    started = _forensic_datetime.now(_forensic_timezone.utc).isoformat()
    header = f'run_id={run_id}\nstage={stage}\nstarted_utc={started}\ncommand={cmd}\n--- output ---\n'
    history_log.write_text(header, encoding='utf-8')
    if canonical_history is not None:
        canonical_history.write_text(header, encoding='utf-8')

    return_code = -1
    caught = None
    completed = None
    process = None
    try:
        process = _forensic_subprocess.Popen(
            cmd,
            shell=isinstance(cmd, str),
            cwd=str(run_cwd),
            stdout=_forensic_subprocess.PIPE,
            stderr=_forensic_subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1,
        )
        with history_log.open('a', encoding='utf-8') as local_handle:
            canonical_handle = (
                canonical_history.open('a', encoding='utf-8')
                if canonical_history is not None
                else None
            )
            try:
                for line in process.stdout or ():
                    print(line, end='', flush=True)
                    local_handle.write(line)
                    local_handle.flush()
                    if canonical_handle is not None:
                        canonical_handle.write(line)
                        canonical_handle.flush()
                return_code = int(process.wait())
                local_handle.flush()
                _forensic_os.fsync(local_handle.fileno())
                if canonical_handle is not None:
                    canonical_handle.flush()
                    _forensic_os.fsync(canonical_handle.fileno())
            finally:
                if canonical_handle is not None:
                    canonical_handle.close()
        completed = _forensic_subprocess.CompletedProcess(cmd, return_code)
    except BaseException as exc:
        caught = exc
        return_code = int(getattr(exc, 'returncode', -1))
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except Exception:
                process.kill()
                process.wait()
    finally:
        footer = (
            '\n--- end ---\n'
            f'ended_utc={_forensic_datetime.now(_forensic_timezone.utc).isoformat()}\n'
            f'return_code={return_code}\n'
        )
        with history_log.open('a', encoding='utf-8') as handle:
            handle.write(footer)
            handle.flush()
            _forensic_os.fsync(handle.fileno())
        if canonical_history is not None:
            # The underlying helper streams to this same canonical history path
            # when supported; append the footer or refresh from the local copy.
            try:
                _forensic_shutil.copy2(history_log, canonical_history)
            except Exception as exc:
                print('WARNING: durable history refresh failed:', exc)
        try:
            _forensic_shutil.copy2(history_log, stable_log)
            if canonical_root is not None:
                try:
                    revision_base = (_ForensicPath(globals().get('REPO_DIR', run_cwd)) / 'reports' / 'revision').resolve()
                    relative = stable_log.resolve().relative_to(revision_base)
                except (ValueError, TypeError):
                    relative = _ForensicPath('logs') / stable_log.name
                canonical_stable = canonical_root / relative
                canonical_stable.parent.mkdir(parents=True, exist_ok=True)
                _forensic_shutil.copy2(history_log, canonical_stable)
        except Exception as exc:
            print('WARNING: latest log refresh failed:', exc)
    print('Run history log:', history_log)
    if canonical_history is not None:
        print('Durable run history log:', canonical_history)
    if caught is not None:
        raise caught
    if check and return_code:
        raise _forensic_subprocess.CalledProcessError(return_code, cmd)
    return completed
'''.lstrip("\n")


def source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def set_source(cell: dict, value: str) -> None:
    cell["source"] = value.splitlines(keepends=True)


def load(name: str) -> dict:
    return json.loads((NOTEBOOK_DIR / name).read_text(encoding="utf-8"))


def save(name: str, notebook: dict) -> None:
    path = NOTEBOOK_DIR / name
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def authority_pin_source(*, bootstrap: bool) -> str:
    bootstrap_literal = "True" if bootstrap else "False"
    return f'''{AUTHORITY_BLOCK_START}
{AUTHORITY_MARKER}
{AUTHORITY_SCHEMA_MARKER}
_AUTHORITY_BOOTSTRAP_ALLOWED = {bootstrap_literal}

def _pin_forensic_code_authority():
    import json as _authority_json
    import os as _authority_os
    import re as _authority_re
    import subprocess as _authority_subprocess
    import uuid as _authority_uuid
    from datetime import datetime as _authority_datetime, timezone as _authority_timezone
    from pathlib import Path as _AuthorityPath

    manifest_path = _AuthorityPath(MIRROR_REVISION_ROOT) / 'manifests' / 'notebook_code_authority.json'
    requested_commit = _authority_os.environ.get('ECG_RAMBA_AUTHORITY_COMMIT', '').strip().lower()
    reset_requested = _authority_os.environ.get('ECG_RAMBA_RESET_CODE_AUTHORITY', '0') == '1'
    commit_pattern = _authority_re.compile(r'[0-9a-f]{{40}}')

    def git(*args, check=True):
        result = _authority_subprocess.run(
            ['git', *args],
            cwd=str(REPO_DIR),
            check=False,
            text=True,
            stdout=_authority_subprocess.PIPE,
            stderr=_authority_subprocess.STDOUT,
        )
        if check and result.returncode:
            raise RuntimeError(
                'Code-authority git command failed: git '
                + ' '.join(args)
                + '\\n'
                + (result.stdout or '')[-4000:]
            )
        return result

    if reset_requested and not _AUTHORITY_BOOTSTRAP_ALLOWED:
        raise RuntimeError(
            'Only Notebook 00 may rotate the canonical code authority. '
            'Run Notebook 00 with ECG_RAMBA_RESET_CODE_AUTHORITY=1 and an explicit '
            'ECG_RAMBA_AUTHORITY_COMMIT.'
        )
    if reset_requested and not commit_pattern.fullmatch(requested_commit):
        raise RuntimeError(
            'Authority reset requires ECG_RAMBA_AUTHORITY_COMMIT as a full 40-character git SHA.'
        )

    manifest = None
    if manifest_path.is_file() and not reset_requested:
        manifest = _authority_json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('capability') != 'canonical_git_commit_pin_v1':
            raise RuntimeError('Canonical code-authority manifest capability is invalid.')
        if int(manifest.get('schema_version', 0)) != 1:
            raise RuntimeError('Canonical code-authority manifest schema is invalid.')
        expected_commit = str(manifest.get('git_commit', '')).strip().lower()
        if not commit_pattern.fullmatch(expected_commit):
            raise RuntimeError('Canonical code-authority manifest lacks a full git SHA.')
        if str(manifest.get('repository_url', '')).rstrip('/') != str(REPO_URL).rstrip('/'):
            raise RuntimeError('Canonical code-authority repository URL differs from this notebook.')
        if str(manifest.get('branch', '')) != str(BRANCH):
            raise RuntimeError('Canonical code-authority branch differs from this notebook runtime.')
        if requested_commit and requested_commit != expected_commit:
            raise RuntimeError(
                'ECG_RAMBA_AUTHORITY_COMMIT differs from the canonical authority manifest. '
                'Rotate authority explicitly in Notebook 00; do not override it in a downstream notebook.'
            )
    else:
        if not _AUTHORITY_BOOTSTRAP_ALLOWED:
            raise FileNotFoundError(
                'Canonical code-authority manifest is missing. Run Notebook 00 first in a fresh runtime; '
                'downstream notebooks fail closed instead of following a moving branch.'
            )
        expected_commit = requested_commit or git('rev-parse', 'HEAD').stdout.strip().lower()
        if not commit_pattern.fullmatch(expected_commit):
            raise RuntimeError('Notebook 00 could not resolve a full code-authority git SHA.')

    tracked_status = git('status', '--porcelain', '--untracked-files=no').stdout.strip()
    if tracked_status:
        raise RuntimeError(
            'Tracked files differ from git before authority checkout. Use a fresh Colab clone; '
            'authority pinning will not stash or overwrite local edits.\\n' + tracked_status[:4000]
        )

    fetch = git('fetch', 'origin', '--prune', check=False)
    if fetch.returncode:
        print('WARNING: git fetch failed; accepting only an already-present pinned commit.')
        print((fetch.stdout or '')[-2000:])
    git('cat-file', '-e', expected_commit + '^{{commit}}')
    git('checkout', '--detach', expected_commit)
    observed_commit = git('rev-parse', 'HEAD').stdout.strip().lower()
    if observed_commit != expected_commit:
        raise RuntimeError(
            f'Code-authority checkout mismatch: expected={{expected_commit}} observed={{observed_commit}}'
        )

    if manifest is None or reset_requested:
        manifest = {{
            'capability': 'canonical_git_commit_pin_v1',
            'schema_version': 1,
            'git_commit': expected_commit,
            'repository_url': str(REPO_URL),
            'branch': str(BRANCH),
            'established_utc': _authority_datetime.now(_authority_timezone.utc).isoformat(),
            'established_by': '00_colab_bootstrap.ipynb',
            'selection': 'explicit_environment_sha' if requested_commit else 'fetched_branch_head_at_initial_bootstrap',
        }}
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = manifest_path.with_name(
            manifest_path.name + '.partial.' + _authority_uuid.uuid4().hex
        )
        with temporary.open('w', encoding='utf-8') as handle:
            handle.write(_authority_json.dumps(manifest, indent=2, sort_keys=True) + '\\n')
            handle.flush()
            _authority_os.fsync(handle.fileno())
        _authority_os.replace(temporary, manifest_path)
        print('Established canonical code authority:', manifest_path)

    _authority_os.environ['ECG_RAMBA_AUTHORITY_COMMIT'] = expected_commit
    globals()['CODE_AUTHORITY_MANIFEST_PATH'] = manifest_path
    globals()['CODE_AUTHORITY'] = manifest
    print('Pinned code authority:', expected_commit)
    print('Authority manifest   :', manifest_path)
    return manifest

CODE_AUTHORITY = _pin_forensic_code_authority()
{AUTHORITY_BLOCK_END}
'''


def remove_managed_authority_block(text: str) -> str:
    pattern = re.compile(
        rf"^[ \t]*{re.escape(AUTHORITY_BLOCK_START)}.*?"
        rf"^[ \t]*{re.escape(AUTHORITY_BLOCK_END)}[ \t]*\n*",
        flags=re.DOTALL | re.MULTILINE,
    )
    return pattern.sub("", text)


def insert_authority_before(text: str, anchor: str, *, bootstrap: bool, indent: str = "") -> str:
    text = remove_managed_authority_block(text)
    position = text.find(anchor)
    if position < 0:
        raise RuntimeError(f"Code-authority insertion anchor not found: {anchor!r}")
    block = textwrap.indent(authority_pin_source(bootstrap=bootstrap).rstrip(), indent)
    prefix = text[:position].rstrip()
    suffix = text[position + len(anchor):].lstrip("\n")
    return prefix + "\n\n" + block + "\n" + indent + anchor.lstrip() + suffix


def insert_authority_after_last(
    text: str,
    anchor: str,
    *,
    bootstrap: bool,
    indent: str = "",
) -> str:
    text = remove_managed_authority_block(text)
    position = text.rfind(anchor)
    if position < 0:
        raise RuntimeError(f"Code-authority insertion anchor not found: {anchor!r}")
    insertion = position + len(anchor)
    block = textwrap.indent(authority_pin_source(bootstrap=bootstrap).rstrip(), indent)
    prefix = text[:insertion].rstrip("\n")
    suffix = text[insertion:].lstrip("\n")
    return prefix + "\n\n" + block + "\n\n" + suffix


def integrate_code_authority() -> None:
    names = (
        "00_colab_bootstrap.ipynb",
        "01_a0_protocol_audit.ipynb",
        "02_predictions_and_external_eval.ipynb",
        "02a_retrain_best_ema.ipynb",
        "03_calibration_and_ci.ipynb",
        "04_baselines_and_component_checks.ipynb",
        "05_hrv_domain_and_robustness.ipynb",
        "06_pooling_and_representation.ipynb",
        "07_results_freeze.ipynb",
    )
    for name in names:
        notebook = load(name)
        setup_cells = [
            cell
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
            and "REPO_URL" in source(cell)
            and "REPO_DIR" in source(cell)
            and "git" in source(cell)
            and ("git clone" in source(cell) or "git pull" in source(cell))
        ]
        expected_count = 2 if name == "07_results_freeze.ipynb" else 1
        if len(setup_cells) != expected_count:
            raise RuntimeError(
                f"{name} code-authority setup candidate_count={len(setup_cells)} expected={expected_count}"
            )
        for cell_index, cell in enumerate(setup_cells):
            text = source(cell)
            if name == "07_results_freeze.ipynb" and "RUN_FULL_MIRROR_RESTORE_07" in text:
                text = text.replace(
                    "RUN_FULL_MIRROR_RESTORE_07\n = os.environ.get('ECG_RAMBA_FULL_MIRROR_RESTORE_07', '0') == '1'",
                    "RUN_FULL_MIRROR_RESTORE_07 = os.environ.get('ECG_RAMBA_FULL_MIRROR_RESTORE_07', '0') == '1'",
                )
                text = insert_authority_before(
                    text,
                    "RUN_FULL_MIRROR_RESTORE_07 = os.environ.get('ECG_RAMBA_FULL_MIRROR_RESTORE_07', '0') == '1'\n",
                    bootstrap=False,
                )
            elif name == "07_results_freeze.ipynb":
                text = insert_authority_before(
                    text,
                    "print('Direct setup guard complete.')\n",
                    bootstrap=False,
                    indent="    ",
                )
            else:
                text = insert_authority_after_last(
                    text,
                    "os.chdir(REPO_DIR)\n",
                    bootstrap=name == "00_colab_bootstrap.ipynb" and cell_index == 0,
                )
            set_source(cell, text)
        save(name, notebook)


def install_run_history(notebook: dict) -> None:
    for cell in notebook.get("cells", []):
        text = source(cell)
        if cell.get("cell_type") != "code" or "def run(" not in text:
            continue
        if RUN_HISTORY_MARKER in text:
            wrapper_start = text.find("# Forensic run-history wrapper.")
            if wrapper_start < 0:
                raise RuntimeError("Run-history marker exists without its wrapper boundary")
            text = text[:wrapper_start].rstrip() + "\n\n" + RUN_HISTORY_WRAPPER
            set_source(cell, text)
            continue
        # The wrapper writes the run header first. Appending lets the underlying
        # helper stream without erasing it and leaves a useful partial log after interruption.
        text = text.replace(".open('w', encoding='utf-8')", ".open('a', encoding='utf-8')")
        text = text.rstrip() + "\n\n" + RUN_HISTORY_WRAPPER
        set_source(cell, text)


def replace_installer_discovery(text: str) -> str:
    pattern = re.compile(
        r"\s*required_markers\s*=\s*\[[^\]]+\]\s*\n\s*if all\(marker in source for marker in required_markers\):",
        flags=re.DOTALL,
    )
    capability_predicate = (
        f"{MAMBA_MARKER!r} in source and {MAMBA_SCHEMA_MARKER!r} in source"
    )
    text = pattern.sub(f"\n            if {capability_predicate}:", text)
    # Normalize the capability predicate from the indentation of its body.  The
    # consumers use this discovery block at different nesting depths, so a
    # hard-coded indent can produce syntactically invalid notebook cells.
    marker_literal = re.escape(repr(MAMBA_MARKER))
    schema_literal = re.escape(repr(MAMBA_SCHEMA_MARKER))
    marker_pattern = re.compile(
        rf"(?m)^[ \t]*if {marker_literal} in source(?: and {schema_literal} in source)?:\n"
        r"(?P<body>[ \t]*)installer_candidates\.append",
    )

    def align_marker(match: re.Match[str]) -> str:
        body_indent = match.group("body")
        if len(body_indent) < 4:
            raise RuntimeError("Mamba installer discovery body is not indented")
        predicate_indent = body_indent[:-4]
        return (
            f"{predicate_indent}if {capability_predicate}:\n"
            f"{body_indent}installer_candidates.append"
        )

    text = marker_pattern.sub(align_marker, text)
    return text


def integrate_notebook02() -> None:
    name = "02_predictions_and_external_eval.ipynb"
    notebook = load(name)
    candidates = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code" and "AUTO_PIN_TORCH_FOR_MAMBA" in source(cell) and "Mamba wheel environment" in source(cell)]
    if len(candidates) != 1:
        raise RuntimeError(f"Notebook 02 Mamba installer candidate_count={len(candidates)}")
    text = source(candidates[0])
    installer_header = MAMBA_MARKER + "\n" + MAMBA_SCHEMA_MARKER + "\n"
    text = text.replace(MAMBA_MARKER + "\n", "")
    text = text.replace(MAMBA_SCHEMA_MARKER + "\n", "")
    set_source(candidates[0], installer_header + text)

    base_candidates = [
        cell
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
        and "INSTALL_BASE_DEPS = True" in source(cell)
        and "REPAIR_BROKEN_NUMERIC_STACK" in source(cell)
    ]
    if len(base_candidates) != 1:
        raise RuntimeError(f"Notebook 02 base installer candidate_count={len(base_candidates)}")
    base_text = source(base_candidates[0])
    base_header = BASE_INSTALLER_MARKER + "\n" + BASE_INSTALLER_SCHEMA_MARKER + "\n"
    base_text = base_text.replace(BASE_INSTALLER_MARKER + "\n", "")
    base_text = base_text.replace(BASE_INSTALLER_SCHEMA_MARKER + "\n", "")
    set_source(base_candidates[0], base_header + base_text)
    strict_oof_flags = (
        "--expected-checkpoint-kind final_ema --manuscript-ready-strict "
        "--group-sidecar reports/revision/manifests/oof_final_ema_group_sidecar.npz"
    )
    for cell in notebook["cells"]:
        text = source(cell)
        legacy_start = "# Notebook 02 may be rerun in a fresh /content clone"
        if NOTEBOOK02_PREFLIGHT_START in text:
            start = text.index(NOTEBOOK02_PREFLIGHT_START)
            end = text.index(NOTEBOOK02_PREFLIGHT_END, start) + len(NOTEBOOK02_PREFLIGHT_END)
            if end < len(text) and text[end] == "\n":
                end += 1
            text = text[:start] + NOTEBOOK02_CAPABILITY_PREFLIGHT + text[end:]
        elif legacy_start in text and "print('Revision compatibility preflight: OK')" in text:
            start = text.index(legacy_start)
            terminal = "print('Revision compatibility preflight: OK')"
            end = text.index(terminal, start) + len(terminal)
            if end < len(text) and text[end] == "\n":
                end += 1
            text = text[:start] + NOTEBOOK02_CAPABILITY_PREFLIGHT + text[end:]
        if (
            "scripts/revision/06_freeze_oof.py" in text
            and "--manuscript-ready-strict" not in text
        ):
            text = text.replace(
                "--expected-checkpoint-kind final_ema",
                strict_oof_flags,
            )
        if "RUN_OOF_EXPORT = True" in text and "def freeze_oof(label):" in text:
            sidecar_anchor = "OOF_FOLD_CACHE_DIR.mkdir(parents=True, exist_ok=True)\n"
            if "def ensure_oof_group_sidecar():" not in text:
                if sidecar_anchor not in text:
                    raise RuntimeError("Notebook 02 OOF sidecar-preflight anchor missing")
                text = text.replace(
                    sidecar_anchor,
                    sidecar_anchor + NOTEBOOK02_OOF_SIDECAR_PREFLIGHT,
                    1,
                )
            text = text.replace(
                "    print('final_ema OOF freeze contract is not ready yet; inference may be required.')",
                "    print('final_ema OOF strict contract is not ready; attempting CPU metadata refresh before any inference decision.')",
            )
            refreshed_ready = r'''oof_ready = False if FORCE_RERUN_OOF else freeze_oof('existing artifacts')
oof_core_available = all(path.is_file() and path.stat().st_size > 0 for path in OOF_CORE_ARTIFACTS)
if not FORCE_RERUN_OOF and not oof_ready and oof_core_available and CANONICAL_GROUP_SIDECAR.is_file():
    print('Refreshing the strict OOF freeze metadata on CPU; existing predictions remain unchanged.')
    freeze_refresh_command = freeze_command + ' --metadata-refresh-from-existing-oof'
    refresh_result = run(
        freeze_refresh_command,
        check=False,
        log_path='reports/revision/logs/oof_final_ema_freeze_refresh.log',
    )
    if refresh_result.returncode == 0:
        freeze_publish_args = ' '.join([
            '--include-path "manifests/oof_final_ema_group_sidecar.npz"',
            '--include-path "manifests/oof_final_ema_freeze_manifest.json"',
        ])
        run(
            f'python -u scripts/revision/artifact_mirror.py publish --verify-existing size '
            f'--source-conflict-policy source {freeze_publish_args} --mirror-root "{stable_mirror}"',
            log_path='reports/revision/logs/oof_freeze_refresh_immediate_mirror_publish.log',
        )
        oof_ready = freeze_oof('after CPU metadata refresh')
    else:
        print('CPU freeze metadata refresh failed; model inference will not be started automatically while complete OOF artifacts exist.')
'''
            ready_pattern = re.compile(
                r"oof_ready = False if FORCE_RERUN_OOF else freeze_oof\('existing artifacts'\)\n"
                r".*?(?=\ncommand = \()",
                flags=re.DOTALL,
            )
            text, ready_replacements = ready_pattern.subn(refreshed_ready.rstrip(), text, count=1)
            if ready_replacements != 1:
                raise RuntimeError(
                    f"Notebook 02 OOF metadata-refresh replacement_count={ready_replacements}"
                )
            legacy_branch = """if RUN_OOF_EXPORT and (FORCE_RERUN_OOF or not oof_ready):
    require_gpu_inference_runtime('Canonical final_ema OOF export')"""
            guarded_branch = """oof_inference_required = bool(FORCE_RERUN_OOF or not oof_core_available)
if RUN_OOF_EXPORT and oof_inference_required:
    require_gpu_inference_runtime('Canonical final_ema OOF export')"""
            if legacy_branch in text:
                text = text.replace(legacy_branch, guarded_branch, 1)
            generation_anchor = "    run(command, log_path='reports/revision/logs/oof_final_ema_generate_predictions.log')\n    run(freeze_command, log_path='reports/revision/logs/oof_final_ema_freeze_validation.log')"
            generation_with_sidecar = "    run(command, log_path='reports/revision/logs/oof_final_ema_generate_predictions.log')\n    ensure_oof_group_sidecar()\n    run(freeze_command, log_path='reports/revision/logs/oof_final_ema_freeze_validation.log')"
            if generation_anchor in text:
                text = text.replace(generation_anchor, generation_with_sidecar, 1)
            legacy_tail = """elif oof_ready and not FORCE_RERUN_OOF:
    print('Skipping final_ema OOF inference because the frozen artifact contract passed.')
else:
    print(f'OOF export disabled. Set RUN_OOF_EXPORT=True to execute: {command}')"""
            guarded_tail = """elif oof_ready and not FORCE_RERUN_OOF:
    print('Skipping final_ema OOF inference because the frozen artifact contract passed.')
elif oof_core_available:
    raise RuntimeError(
        'Complete OOF prediction artifacts are present, but their strict provenance/freeze contract failed. '
        'GPU inference was intentionally not started. Review oof_final_ema_freeze_refresh.log and repair the '
        'specific sidecar, split-membership, or freeze metadata blocker; set FORCE_RERUN_OOF=True only when the '
        'prediction artifact itself is proven invalid.'
    )
else:
    print(f'OOF export disabled. Set RUN_OOF_EXPORT=True to execute: {command}')"""
            if legacy_tail in text:
                text = text.replace(legacy_tail, guarded_tail, 1)
        if (
            "oof_final_ema_freeze_manifest.json" in text
            and "expected = [" in text
            and "oof_final_ema_group_sidecar.npz" not in text
        ):
            text = text.replace(
                "    Path('reports/revision/manifests/oof_final_ema_freeze_manifest.json'),\n",
                "    Path('reports/revision/manifests/oof_final_ema_freeze_manifest.json'),\n"
                "    Path('reports/revision/manifests/oof_final_ema_group_sidecar.npz'),\n",
                1,
            )
        if "def external_prediction_ready(dataset):" in text:
            pattern = re.compile(
                r"def external_prediction_ready\(dataset\):\n.*?(?=\n\ndef resolve_auto_flag)",
                flags=re.DOTALL,
            )
            text, replacement_count = pattern.subn(
                NOTEBOOK02_EXTERNAL_REUSE_FUNCTION.rstrip(),
                text,
            )
            if replacement_count != 1:
                raise RuntimeError(
                    f"Notebook 02 external reuse function replacement_count={replacement_count}"
                )
            text = text.replace(
                "artifact_mirror.py publish --verify-existing size --mirror-root",
                "artifact_mirror.py publish --verify-existing size --source-conflict-policy source --mirror-root",
            )
            broad_dataset_publish = """            run(
                f'python -u scripts/revision/artifact_mirror.py publish --verify-existing size --source-conflict-policy source --mirror-root "{mirror_root}"',
                log_path=f'reports/revision/logs/{dataset}_external_export_mirror_publish.log',
            )"""
            selected_dataset_publish = """            external_publish_paths = [
                *external_required_artifacts(dataset),
                Path('reports/revision/experimental/external/external_summary_experimental.csv'),
            ]
            if dataset == 'georgia':
                external_publish_paths.append(GEORGIA_CODE_INVENTORY_OUT)
            elif dataset == 'cpsc2021':
                external_publish_paths.append(CPSC_ANNOTATION_AUDIT_OUT)
            external_publish_args = ' '.join(
                f'--include-path "{path.relative_to(Path("reports/revision")).as_posix()}"'
                for path in external_publish_paths
            )
            run(
                f'python -u scripts/revision/artifact_mirror.py publish --verify-existing size '
                f'--source-conflict-policy source {external_publish_args} --mirror-root "{mirror_root}"',
                log_path=f'reports/revision/logs/{dataset}_external_export_mirror_publish.log',
            )"""
            if broad_dataset_publish in text:
                text = text.replace(broad_dataset_publish, selected_dataset_publish, 1)
            broad_final_publish = """if external_export_ran:
    print('All successful external datasets were already published individually; refreshing the merged manifest.')
    mirror_root = globals().get('stable_mirror', DRIVE_ROOT / 'revision_artifacts' / 'reports' / 'revision')
    run(
        f'python -u scripts/revision/artifact_mirror.py publish --verify-existing size --source-conflict-policy source --mirror-root "{mirror_root}"',
        log_path='reports/revision/logs/external_export_immediate_mirror_publish.log',
    )"""
            selected_final_publish = """if external_export_ran:
    print('Successful external exports were published with exact source-path selection; no broad mirror overwrite is needed.')"""
            if broad_final_publish in text:
                text = text.replace(broad_final_publish, selected_final_publish, 1)
        if "EXTERNAL_GATE_INPUT_PATHS = [" in text:
            gate_list_end = """        Path(f'reports/revision/experimental/external/{dataset}/{dataset}_full_prediction_run_manifest.json'),
    ]
]"""
            gate_list_replacement = gate_list_end + """
EXTERNAL_GATE_INPUT_PATHS.extend([
    Path('reports/revision/tables/table_georgia_snomed_code_inventory.csv'),
    Path('reports/revision/tables/table_cpsc2021_annotation_audit.csv'),
])"""
            if "table_georgia_snomed_code_inventory.csv" not in text:
                if gate_list_end not in text:
                    raise RuntimeError("Notebook 02 external gate input-list anchor missing")
                text = text.replace(gate_list_end, gate_list_replacement, 1)
            gate_contract_extension = """EXTERNAL_GATE_INPUT_PATHS.extend([
    Path('reports/revision/predictions/oof_final_ema_predictions.npz'),
    Path('reports/revision/manifests/oof_final_ema_freeze_manifest.json'),
    Path('reports/revision/manifests/oof_final_ema_prediction_run_manifest.json'),
])"""
            if "reports/revision/predictions/oof_final_ema_predictions.npz" not in text:
                audit_extension = """EXTERNAL_GATE_INPUT_PATHS.extend([
    Path('reports/revision/tables/table_georgia_snomed_code_inventory.csv'),
    Path('reports/revision/tables/table_cpsc2021_annotation_audit.csv'),
])"""
                if audit_extension not in text:
                    raise RuntimeError("Notebook 02 external gate audit-input anchor missing")
                text = text.replace(
                    audit_extension,
                    audit_extension + "\n" + gate_contract_extension,
                    1,
                )
            text = text.replace("total=15.", "total={len(EXTERNAL_GATE_INPUT_PATHS)}.")
            text = text.replace(
                "External gate input contract: all 15 artifacts are present in the active repo.",
                "External gate input contract: all {len(EXTERNAL_GATE_INPUT_PATHS)} artifacts are present in the active repo.",
            )
            text = text.replace(
                "print('External gate input contract: all {len(EXTERNAL_GATE_INPUT_PATHS)} artifacts are present in the active repo.')",
                "print(f'External gate input contract: all {len(EXTERNAL_GATE_INPUT_PATHS)} artifacts are present in the active repo.')",
            )
            source_preflight_anchor = (
                "print(f'External gate input contract: all {len(EXTERNAL_GATE_INPUT_PATHS)} "
                "artifacts are present in the active repo.')\n\n"
            )
            if "External gate source-bound preflight:" not in text:
                source_preflight = r'''# Fail before the long bootstrap when restored predictions were produced by a
# stale exporter/protocol/archive. This validator is CPU-only and has no Mamba/WFDB dependency.
from scripts.revision.external_reuse_contract import validate_external_prediction_reuse

gate_archive_names = {
    'ptbxl': ['PTB-XL.zip', 'ptb-xl.zip', 'ptbxl.zip'],
    'georgia': ['Georgia.zip'],
    'cpsc2021': ['cpsc2021.zip', 'CPSC2021.zip'],
}
gate_source_preflight = {}
for dataset in ['ptbxl', 'georgia', 'cpsc2021']:
    archive = next(
        (DRIVE_ROOT / name for name in gate_archive_names[dataset] if (DRIVE_ROOT / name).is_file()),
        None,
    )
    result = validate_external_prediction_reuse(
        dataset,
        revision_root=Path('reports/revision'),
        archive_path=archive,
        exporter_path=Path('scripts/revision/03_generate_external_predictions.py'),
        oof_path=Path('reports/revision/predictions/oof_final_ema_predictions.npz'),
        freeze_path=Path('reports/revision/manifests/oof_final_ema_freeze_manifest.json'),
        archive_hash_cache_dir=gate_restore_root / 'manifests' / 'external_archive_hash_cache',
        threshold=0.5,
        q=3.0,
    )
    gate_source_preflight[dataset] = result
    print(
        f'External gate source-bound preflight: dataset={dataset} '
        f'ready={result.get("ready", False)} reasons={result.get("reasons", [])}'
    )
stale_gate_inputs = {
    dataset: result.get('reasons', [])
    for dataset, result in gate_source_preflight.items()
    if not result.get('ready', False)
}
if stale_gate_inputs:
    details = ' ; '.join(
        f'{dataset}=' + ','.join(reasons)
        for dataset, reasons in stale_gate_inputs.items()
    )
    raise RuntimeError(
        'External protocol gate stopped before bootstrap because restored prediction artifacts are stale: '
        + details
        + '. Reconnect an A100 High-RAM runtime and run Experimental External Prediction Commands '
        'with RUN_PTBXL_EXPORT=RUN_GEORGIA_EXPORT=RUN_CPSC2021_EXPORT="auto". '
        'Wait for External cache handoff: VERIFIED before returning to this CPU gate cell.'
    )

'''
                if source_preflight_anchor not in text:
                    raise RuntimeError("Notebook 02 gate source-preflight anchor missing")
                text = text.replace(
                    source_preflight_anchor,
                    source_preflight_anchor + source_preflight,
                    1,
                )
            text = text.replace(
                "artifact_mirror.py publish --verify-existing size --mirror-root",
                "artifact_mirror.py publish --verify-existing size --source-conflict-policy source --mirror-root",
            )
            gate_run_anchor = "if RUN_EXTERNAL_PROTOCOL_GATE:\n"
            if "gate_publish_args = ' '.join(" not in text:
                gate_publish_setup = """GATE_OUTPUT_PATHS = [
    Path('reports/revision/metrics/external_protocol_gate_summary.csv'),
    *[
        path
        for dataset in ['ptbxl', 'georgia', 'cpsc2021']
        for path in [
            Path(f'reports/revision/metrics/external_{dataset}_protocol_gate.json'),
            Path(f'reports/revision/tables/table_external_{dataset}_label_mapping.csv'),
            Path(f'reports/revision/tables/table_external_{dataset}_metrics.csv'),
            Path(f'reports/revision/manifests/external_{dataset}_protocol_gate_manifest.json'),
        ]
    ],
]
gate_publish_args = ' '.join(
    f'--include-path "{path.relative_to(Path("reports/revision")).as_posix()}"'
    for path in GATE_OUTPUT_PATHS
)

"""
                if gate_run_anchor not in text:
                    raise RuntimeError("Notebook 02 external gate run anchor missing")
                text = text.replace(gate_run_anchor, gate_publish_setup + gate_run_anchor, 1)
            broad_gate_publish = (
                "f'python -u scripts/revision/artifact_mirror.py publish --verify-existing size "
                "--source-conflict-policy source --mirror-root \"{gate_restore_root}\"'"
            )
            selected_gate_publish = (
                "f'python -u scripts/revision/artifact_mirror.py publish --verify-existing size "
                "--source-conflict-policy source {gate_publish_args} --mirror-root \"{gate_restore_root}\"'"
            )
            text = text.replace(broad_gate_publish, selected_gate_publish, 1)
        set_source(cell, text)
    install_run_history(notebook)
    save(name, notebook)


def integrate_notebook00() -> None:
    name = "00_colab_bootstrap.ipynb"
    notebook = load(name)
    for cell in notebook["cells"]:
        text = source(cell)
        if "INSTALL_MAMBA_IN_NOTEBOOK00" in text and "installer_text" in text:
            text = text.replace(
                f"if {MAMBA_MARKER!r} in installer_text:",
                f"if {MAMBA_MARKER!r} in installer_text and {MAMBA_SCHEMA_MARKER!r} in installer_text:",
            )
            text = text.replace(
                "expected exactly one capability marker.",
                "expected exactly one capability/schema marker pair.",
            )
            set_source(cell, text)
        if "This Colab runtime is CPU-only" in text and "torch.cuda.is_available()" in text:
            text = re.sub(
                r"if torch\.cuda\.is_available\(\):\n\s+print\('GPU   :'.*?\nelse:\n\s+raise RuntimeError\(.*?\n\s+\)",
                "if torch.cuda.is_available():\n    print('GPU   :', torch.cuda.get_device_name(0))\nelse:\n    print('CPU audit mode: Notebook 00/01 can complete without CUDA. GPU is required only by later inference/training cells.')",
                text,
                flags=re.DOTALL,
            )
            set_source(cell, text)
        if "Could not locate the canonical Mamba installer cell in notebook 02" in text:
            replacement = f'''import json\nimport os\nfrom pathlib import Path\n\nINSTALL_MAMBA_IN_NOTEBOOK00 = os.environ.get('ECG_RAMBA_INSTALL_MAMBA_IN_NOTEBOOK00', '0') == '1'\nif not INSTALL_MAMBA_IN_NOTEBOOK00:\n    print('Skipping Mamba installation in Notebook 00. CPU storage/protocol audit is complete; Notebook 02 installs Mamba only when GPU inference is required.')\nelse:\n    import torch\n    if not torch.cuda.is_available():\n        raise RuntimeError('ECG_RAMBA_INSTALL_MAMBA_IN_NOTEBOOK00=1 requires a CUDA runtime.')\n    installer_path = REPO_DIR / 'notebooks' / '02_predictions_and_external_eval.ipynb'\n    installer_nb = json.loads(installer_path.read_text(encoding='utf-8'))\n    candidates = []\n    for cell_index, installer_cell in enumerate(installer_nb['cells']):\n        if installer_cell.get('cell_type') != 'code':\n            continue\n        installer_text = ''.join(installer_cell.get('source', []))\n        if {MAMBA_MARKER!r} in installer_text and {MAMBA_SCHEMA_MARKER!r} in installer_text:\n            candidates.append((cell_index, installer_text))\n    if len(candidates) != 1:\n        raise RuntimeError(f'Canonical Mamba installer candidate_count={{len(candidates)}}; expected exactly one capability/schema marker pair.')\n    print('Running canonical Mamba installer from Notebook 02 cell', candidates[0][0])\n    exec(compile(candidates[0][1], str(installer_path) + ':model-deps', 'exec'), globals(), globals())\n'''
            set_source(cell, replacement)
    install_run_history(notebook)
    save(name, notebook)


def integrate_notebook01() -> None:
    name = "01_a0_protocol_audit.ipynb"
    notebook = load(name)
    candidates = [
        cell
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code" and "def run_logged(" in source(cell)
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"Notebook 01 run helper candidate_count={len(candidates)}")
    text = source(candidates[0])
    if RUN_HISTORY_MARKER not in text:
        text = text.replace(".open('w', encoding='utf-8')", ".open('a', encoding='utf-8')")
        text = (
            text.rstrip()
            + "\n\n# Route the Notebook 01 legacy helper through the common forensic history wrapper.\n"
            + "run = run_logged\n\n"
            + RUN_HISTORY_WRAPPER
            + "\nrun_logged = run\n"
        )
        set_source(candidates[0], text)
    save(name, notebook)


def integrate_installer_consumers() -> None:
    for name in ("02a_retrain_best_ema.ipynb", "05_hrv_domain_and_robustness.ipynb", "06_pooling_and_representation.ipynb"):
        notebook = load(name)
        for cell in notebook["cells"]:
            text = source(cell)
            if name == "02a_retrain_best_ema.ipynb" and "def canonical_installer_source(*markers):" in text:
                function_pattern = re.compile(
                    r"def canonical_installer_source\(\*markers\):\n.*?"
                    r"raise RuntimeError\(f'Could not locate canonical installer cell containing: \{markers\}'\)\n",
                    flags=re.DOTALL,
                )
                strict_selector = '''def canonical_installer_source(capability_marker, schema_marker):
    candidates = []
    for notebook_cell in installer_nb['cells']:
        if notebook_cell.get('cell_type') != 'code':
            continue
        installer_source = ''.join(notebook_cell.get('source', []))
        if capability_marker in installer_source and schema_marker in installer_source:
            candidates.append(installer_source)
    if len(candidates) != 1:
        raise RuntimeError(
            f'Canonical installer candidate_count={len(candidates)} for '
            f'capability={capability_marker!r} schema={schema_marker!r}; expected exactly one.'
        )
    return candidates[0]
'''
                text, count = function_pattern.subn(strict_selector, text, count=1)
                if count != 1:
                    raise RuntimeError("Notebook 02a canonical installer selector was not replaced")
            if name == "02a_retrain_best_ema.ipynb" and "canonical_installer_source(" in text:
                text = re.sub(
                    r"base_installer = canonical_installer_source\([^\n]+\)",
                    f"base_installer = canonical_installer_source({BASE_INSTALLER_MARKER!r}, {BASE_INSTALLER_SCHEMA_MARKER!r})",
                    text,
                    count=1,
                )
                text = re.sub(
                    r"model_installer = canonical_installer_source\([^\n]+\)",
                    f"model_installer = canonical_installer_source({MAMBA_MARKER!r}, {MAMBA_SCHEMA_MARKER!r})",
                    text,
                    count=1,
                )
            if "installer_candidates" in text and (
                "AUTO_PIN_TORCH_FOR_MAMBA" in text
                or MAMBA_MARKER in text
                or "required_markers" in text
            ):
                text = replace_installer_discovery(text)
            set_source(cell, text)
        install_run_history(notebook)
        save(name, notebook)


def integrate_notebook02a_training_log() -> None:
    name = "02a_retrain_best_ema.ipynb"
    notebook = load(name)
    candidates = [
        cell
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
        and "RUN_RETRAIN" in source(cell)
        and "scripts/train.py" in source(cell)
        and "existing_ckpts" in source(cell)
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"Notebook 02a training cell candidate_count={len(candidates)}")
    training_source = '''from pathlib import Path
import os
import shutil
import sys

FORENSIC_RETRAIN_STREAMING_LOG_CAPABILITY = 'stage_run_id_durable_stream_v1'
RUN_RETRAIN = os.environ.get('ECG_RAMBA_RUN_RETRAIN', '0') == '1'
RESUME_TRAINING = os.environ.get('ECG_RAMBA_RESUME_TRAINING', '1') == '1'
os.environ['ECG_RAMBA_RESUME_TRAINING'] = '1' if RESUME_TRAINING else '0'

log_path = Path('reports/revision/logs/retrain_best_ema_train.log')
durable_model_log_path = MODEL_DIR / 'retrain_best_ema_train.log'
training_command = f'"{sys.executable}" -u scripts/train.py'
print('Training log (latest local):', log_path)
print('Training logs (durable run history):', MIRROR_REVISION_ROOT / 'logs' / 'history' / 'retrain_best_ema_train')
print('Training log (model-run convenience copy):', durable_model_log_path)
existing_ckpts = sorted(MODEL_DIR.glob('fold*_*.pt'))
if RUN_RETRAIN and existing_ckpts and not RESUME_TRAINING:
    preview = ', '.join(path.name for path in existing_ckpts[:8])
    raise RuntimeError(
        f'Model run directory already contains {len(existing_ckpts)} checkpoint files: {MODEL_DIR}. '
        f'Preview: {preview}. Enable RESUME_TRAINING or select a new ECG_RAMBA_RETRAIN_RUN_ID.'
    )
if existing_ckpts and RESUME_TRAINING:
    print(f'Resume enabled: {len(existing_ckpts)} existing checkpoint files will be audited by scripts/train.py.')
print('$', training_command)

if RUN_RETRAIN:
    run(
        training_command,
        cwd=REPO_DIR,
        log_path=log_path,
    )
    if not log_path.is_file() or log_path.stat().st_size == 0:
        raise RuntimeError('Retraining completed without a non-empty latest-run log.')
    durable_model_log_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_model_log = durable_model_log_path.with_suffix(
        durable_model_log_path.suffix + '.partial'
    )
    shutil.copy2(log_path, temporary_model_log)
    os.replace(temporary_model_log, durable_model_log_path)
    print('Retraining finished:', log_path)
    print('Model-run log refreshed:', durable_model_log_path)
else:
    print(
        'RUN_RETRAIN=False; skipping training and leaving existing artifacts untouched. '
        'Set ECG_RAMBA_RUN_RETRAIN=1 only when retraining is intended.'
    )
'''
    set_source(candidates[0], training_source)
    save(name, notebook)


def integrate_remaining_run_history() -> None:
    for name in (
        "03_calibration_and_ci.ipynb",
        "04_baselines_and_component_checks.ipynb",
        "07_results_freeze.ipynb",
    ):
        notebook = load(name)
        install_run_history(notebook)
        save(name, notebook)


def integrate_notebook03_strict_inputs() -> None:
    name = "03_calibration_and_ci.ipynb"
    notebook = load(name)
    for cell in notebook["cells"]:
        text = source(cell)
        if "CALIBRATION_RESTORE_RELATIVE_PATHS" in text:
            sidecar_entry = "    'manifests/oof_final_ema_group_sidecar.npz',\n"
            if sidecar_entry not in text:
                text = text.replace(
                    "    'manifests/oof_final_ema_freeze_manifest.json',\n",
                    "    'manifests/oof_final_ema_freeze_manifest.json',\n" + sidecar_entry,
                    1,
                )
            text = text.replace(
                "--expected-records 44186 --expected-folds 5 --q 3 --expected-checkpoint-kind final_ema --check-only'",
                "--expected-records 44186 --expected-folds 5 --q 3 --expected-checkpoint-kind final_ema "
                "--manuscript-ready-strict --group-sidecar reports/revision/manifests/oof_final_ema_group_sidecar.npz "
                "--check-existing-freeze'",
            )
        if "FORCE_RERUN_CALIBRATION_CI" in text and "freeze_payload = json.loads" in text:
            anchor = "freeze_payload = json.loads(freeze.read_text(encoding='utf-8'))\n"
            binding = '''freeze_group_sidecar = (freeze_payload.get('group_contract') or {}).get('sidecar') or {}
freeze_group_sidecar_path = Path(str(freeze_group_sidecar.get('path', '')))
if freeze_group_sidecar_path and not freeze_group_sidecar_path.is_absolute():
    freeze_group_sidecar_path = Path.cwd() / freeze_group_sidecar_path
if not freeze_group_sidecar_path.is_file():
    raise FileNotFoundError('Authenticated group sidecar is missing: ' + str(freeze_group_sidecar_path))
freeze_group_sidecar_sha = _sha256(freeze_group_sidecar_path)
if freeze_group_sidecar_sha != freeze_group_sidecar.get('sha256'):
    raise RuntimeError('Authenticated group sidecar SHA differs from the strict freeze manifest.')
'''
            if "freeze_group_sidecar_sha = _sha256" not in text:
                # _sha256 is defined immediately below the original anchor, so
                # insert the sidecar check after that helper definition instead.
                helper_end = "    return digest.hexdigest()\n\n"
                text = text.replace(helper_end, helper_end + binding, 1)
            comparison_anchor = "        if not bootstrap_contract.get('group_sidecar_sha256'):\n            stale_reasons.append('bootstrap_group_sidecar_sha256_missing')\n"
            if comparison_anchor in text and "bootstrap_group_sidecar_sha_mismatch" not in text:
                text = text.replace(
                    comparison_anchor,
                    comparison_anchor
                    + "        elif bootstrap_contract.get('group_sidecar_sha256') != freeze_group_sidecar_sha:\n"
                    + "            stale_reasons.append('bootstrap_group_sidecar_sha_mismatch')\n",
                    1,
                )
        set_source(cell, text)
    save(name, notebook)


def normalize_bootstrap_contracts() -> None:
    replacements = {
        "chapman_record_subject": AUTHENTICATED_BOOTSTRAP_UNIT,
        "chapman_record_one_record_per_subject": AUTHENTICATED_BOOTSTRAP_UNIT,
        "one_chapman_record_per_subject": GROUP_SEMANTICS,
    }
    for name in (
        "03_calibration_and_ci.ipynb",
        "05_hrv_domain_and_robustness.ipynb",
        "07_results_freeze.ipynb",
    ):
        notebook = load(name)
        for cell in notebook["cells"]:
            text = source(cell)
            for old, new in replacements.items():
                text = text.replace(old, new)
            if name == "03_calibration_and_ci.ipynb":
                marker = "FORENSIC_BOOTSTRAP_BINDING_CHECK = 'freeze_group_sidecar_v1'"
                anchor = "        bootstrap_contract = existing.get('bootstrap') or {}\n"
                if anchor in text and marker not in text:
                    text = text.replace(
                        anchor,
                        anchor
                        + f"        {marker}\n"
                        + f"        if bootstrap_contract.get('group_semantics_reference') != {GROUP_REFERENCE!r}:\n"
                        + "            stale_reasons.append('bootstrap_group_semantics_reference_missing_or_mismatched')\n"
                        + "        if not bootstrap_contract.get('group_sidecar_sha256'):\n"
                        + "            stale_reasons.append('bootstrap_group_sidecar_sha256_missing')\n",
                        1,
                    )
            if name == "05_hrv_domain_and_robustness.ipynb":
                marker = "FORENSIC_BOOTSTRAP_BINDING_CHECK = 'freeze_group_sidecar_v1'"
                unit_anchor = f"        expected_bootstrap_unit = {AUTHENTICATED_BOOTSTRAP_UNIT!r}\n"
                if unit_anchor in text and marker not in text:
                    text = text.replace(
                        unit_anchor,
                        f"        {marker}\n" + unit_anchor,
                        1,
                    )
                semantics_line = (
                    f"                or independence.get('independence_contract') != {GROUP_SEMANTICS!r}\n"
                )
                if semantics_line in text and "group_semantics_reference') !=" not in text:
                    text = text.replace(
                        semantics_line,
                        semantics_line
                        + f"                or independence.get('group_semantics_reference') != {GROUP_REFERENCE!r}\n"
                        + "                or not independence.get('group_sidecar_sha256')\n",
                    )
            if name == "07_results_freeze.ipynb":
                marker = "FORENSIC_BOOTSTRAP_BINDING_CHECK = 'freeze_group_sidecar_v1'"
                anchor = "calibration_refresh_reasons = []\n"
                if anchor in text and marker not in text:
                    text = text.replace(
                        anchor,
                        f"{marker}\nEXPECTED_GROUP_REFERENCE = {GROUP_REFERENCE!r}\n" + anchor,
                        1,
                    )
                    preflight_anchor = "if calibration_refresh_reasons:\n"
                    text = text.replace(
                        preflight_anchor,
                        "if calibration_bootstrap.get('group_semantics_reference') != EXPECTED_GROUP_REFERENCE:\n"
                        "    calibration_refresh_reasons.append('bootstrap.group_semantics_reference_missing_or_mismatched')\n"
                        "if not calibration_bootstrap.get('group_sidecar_sha256'):\n"
                        "    calibration_refresh_reasons.append('bootstrap.group_sidecar_sha256_missing')\n"
                        + preflight_anchor,
                        1,
                    )
                post_refresh_anchor = (
                    f"        or calibration_bootstrap.get('independence_contract') != {GROUP_SEMANTICS!r}\n"
                )
                if post_refresh_anchor in text and "or not calibration_bootstrap.get('group_sidecar_sha256')" not in text:
                    text = text.replace(
                        post_refresh_anchor,
                        post_refresh_anchor
                        + "        or calibration_bootstrap.get('group_semantics_reference') != EXPECTED_GROUP_REFERENCE\n"
                        + "        or not calibration_bootstrap.get('group_sidecar_sha256')\n",
                        1,
                    )
                text = text.replace(
                    f"    {GROUP_SEMANTICS!r},\n",
                    "    'PRESENTATION_CONTRACT_SCHEMA_VERSION = 2',\n"
                    "    'AUTHENTICATED_RECORD_BOOTSTRAP_UNIT',\n"
                    "    'CHAPMAN_GROUP_SEMANTICS',\n",
                )
            set_source(cell, text)
        save(name, notebook)


def integrate_notebook07_final_gate() -> None:
    name = "07_results_freeze.ipynb"
    notebook = load(name)
    gate_marker = "FORENSIC_NOTEBOOK07_FINAL_GATE = 'strict_full_sha_authority_update_v3'"
    target = None
    for cell in notebook["cells"]:
        text = source(cell)
        if "final_artifact_inventory.log" in text and (
            "final_evidence_mirror_publish.log" in text
            or "FORENSIC_NOTEBOOK07_FINAL_GATE" in text
        ):
            target = cell
            break
    if target is None:
        raise RuntimeError("Notebook 07 final inventory/publish cell not found")
    gate_source = f'''{gate_marker}
if CODE_AUTHORITY.get('git_commit') != os.environ.get('ECG_RAMBA_AUTHORITY_COMMIT'):
    raise RuntimeError('Notebook 07 code-authority session contract is missing or inconsistent.')
run(
    'python -u scripts/revision/05_artifact_inventory.py',
    log_path='reports/revision/logs/final_artifact_inventory.log',
)
# Current-authority outputs may legitimately replace an older canonical audit.
# The detached commit pin above prevents a stale branch/runtime from publishing.
run(
    f'python -u scripts/revision/artifact_mirror.py publish --verify-existing full --source-conflict-policy newer --mirror-root "{{MIRROR_REVISION_ROOT}}"',
    log_path='reports/revision/logs/final_evidence_authority_publish.log',
)
run(
    f'python -u scripts/revision/38_pipeline_storage_audit.py --canonical-root "{{MIRROR_REVISION_ROOT}}" --strict --full-sha',
    log_path='reports/revision/logs/final_pipeline_storage_audit_strict_full_sha.log',
)
run(
    f'python -u scripts/revision/47_forensic_notebook_audit.py --canonical-root "{{MIRROR_REVISION_ROOT}}" --strict',
    log_path='reports/revision/logs/final_notebook_forensic_audit.log',
)
run(
    f'python -u scripts/revision/artifact_mirror.py publish --verify-existing full --source-conflict-policy newer --mirror-root "{{MIRROR_REVISION_ROOT}}"',
    log_path='reports/revision/logs/final_forensic_audit_authority_publish.log',
)
run(
    f'python -u scripts/revision/38_pipeline_storage_audit.py --canonical-root "{{MIRROR_REVISION_ROOT}}" --strict --full-sha',
    log_path='reports/revision/logs/final_pipeline_storage_audit_post_publish_strict_full_sha.log',
)
'''
    set_source(target, gate_source)

    ledger_cell = next(
        (cell for cell in notebook["cells"] if "presentation_paths = [" in source(cell)),
        None,
    )
    if ledger_cell is not None:
        ledger_text = source(ledger_cell)
        copy_start = ledger_text.find("final_evidence_dir = DRIVE_ROOT / 'final_evidence_tables'")
        copy_end_token = "        print('Copied:', destination)\n"
        copy_end = ledger_text.find(copy_end_token, copy_start)
        if copy_start >= 0 and copy_end >= 0:
            ledger_text = ledger_text[:copy_start] + ledger_text[copy_end + len(copy_end_token):]
            set_source(ledger_cell, ledger_text)

    export_cell = next(
        (cell for cell in notebook["cells"] if "FINAL_TABLE_EXPORT_DIR" in source(cell) and "optional_sources = [" in source(cell)),
        None,
    )
    if export_cell is None:
        raise RuntimeError("Notebook 07 final export cell not found")
    export_text = source(export_cell)
    forensic_sources = [
        "Path('reports/revision/audits/notebook_forensic_audit.md')",
        "Path('reports/revision/tables/table_notebook_cell_audit.csv')",
        "Path('reports/revision/tables/table_reviewer_traceability.csv')",
        "Path('reports/revision/tables/table_statistical_oracle_check.csv')",
        "Path('reports/revision/tables/table_paired_inference_audit.csv')",
        "Path('reports/revision/tables/table_comparator_contract.csv')",
        "Path('reports/revision/tables/table_forensic_rerun_dependencies.csv')",
        "Path('reports/revision/metrics/artifact_provenance_audit.json')",
        "Path('reports/revision/tables/table_hypothesis_control_finding_claim_boundary.csv')",
        "Path('reports/revision/tables/table_hypothesis_control_finding_claim_boundary.tex')",
        "Path('reports/revision/metrics/hypothesis_control_claim_boundary.json')",
        "Path('reports/revision/metrics/matched_oof_calibration_summary.json')",
        "Path('reports/revision/metrics/matched_oof_calibration_bootstrap.json')",
        "Path('reports/revision/tables/table_matched_oof_calibration.csv')",
        "Path('reports/revision/tables/table_matched_oof_calibration_coefficients.csv')",
        "Path('reports/revision/tables/table_matched_oof_calibration.tex')",
        "Path('reports/revision/tables/table_paired_matched_oof_calibration.csv')",
        "Path('reports/revision/figures/figure_matched_calibration_audit.png')",
        "Path('reports/revision/metrics/structured_ablation_5fold_summary.json')",
        "Path('reports/revision/tables/table_structured_ablation_5fold.csv')",
        "Path('reports/revision/tables/table_structured_ablation_5fold.tex')",
        "Path('reports/revision/tables/table_paired_structured_ablation_5fold.csv')",
        "Path('reports/revision/tables/table_true_fewshot_head_ptbxl_learning_curve.csv')",
        "Path('reports/revision/figures/figure_true_fewshot_head_ptbxl_learning_curve.png')",
        "Path('reports/revision/metrics/physiological_interval_probe_summary.json')",
        "Path('reports/revision/tables/table_physiological_interval_probe.csv')",
        "Path('reports/revision/tables/table_physiological_interval_probe_contrasts.csv')",
        "Path('reports/revision/tables/table_physiological_interval_probe.tex')",
    ]
    anchor = "optional_sources = [\n"
    missing_forensic_sources = [item for item in forensic_sources if item not in export_text]
    if missing_forensic_sources:
        export_text = export_text.replace(
            anchor,
            anchor + "    " + ",\n    ".join(missing_forensic_sources) + ",\n",
            1,
        )
        set_source(export_cell, export_text)
    save(name, notebook)


def validate() -> None:
    notebook02 = load("02_predictions_and_external_eval.ipynb")
    marker_count = sum(MAMBA_MARKER in source(cell) for cell in notebook02["cells"])
    if marker_count != 1:
        raise RuntimeError(f"Mamba capability marker count={marker_count}, expected 1")
    schema_marker_count = sum(MAMBA_SCHEMA_MARKER in source(cell) for cell in notebook02["cells"])
    if schema_marker_count != 1:
        raise RuntimeError(f"Mamba schema marker count={schema_marker_count}, expected 1")
    base_marker_count = sum(BASE_INSTALLER_MARKER in source(cell) for cell in notebook02["cells"])
    base_schema_count = sum(BASE_INSTALLER_SCHEMA_MARKER in source(cell) for cell in notebook02["cells"])
    if base_marker_count != 1 or base_schema_count != 1:
        raise RuntimeError(
            "Notebook 02 base installer must contain exactly one stable capability/schema marker pair"
        )
    notebook02_text = "\n".join(source(cell) for cell in notebook02["cells"])
    for token in (
        "--manuscript-ready-strict",
        "oof_final_ema_group_sidecar.npz",
        NOTEBOOK02_PREFLIGHT_START,
        "NOTEBOOK_02_EXTERNAL_EXPORT_CAPABILITY",
        "NOTEBOOK_02_EXTERNAL_GATE_CAPABILITY",
        "_compat_ast.parse",
    ):
        if token not in notebook02_text:
            raise RuntimeError(f"Notebook 02 strict OOF contract token missing: {token}")
    for forbidden in (
        "raw.githubusercontent.com/BrianNguyen29/ECG-RAMBA",
        "annotation_aligned_nonoverlapping_10s_windows_majority_af_or_normal",
        "GATE_SCHEMA_VERSION = 4",
    ):
        if forbidden in notebook02_text:
            raise RuntimeError(f"Notebook 02 stale/mutable compatibility token remains: {forbidden}")
    for name in (
        "00_colab_bootstrap.ipynb",
        "01_a0_protocol_audit.ipynb",
        "02_predictions_and_external_eval.ipynb",
        "02a_retrain_best_ema.ipynb",
        "03_calibration_and_ci.ipynb",
        "04_baselines_and_component_checks.ipynb",
        "05_hrv_domain_and_robustness.ipynb",
        "06_pooling_and_representation.ipynb",
        "07_results_freeze.ipynb",
    ):
        notebook = load(name)
        text = "\n".join(source(cell) for cell in notebook["cells"])
        if RUN_HISTORY_MARKER not in text:
            raise RuntimeError(f"Run-history capability missing from {name}")
        authority_count = text.count(AUTHORITY_MARKER)
        expected_authority_count = 2 if name == "07_results_freeze.ipynb" else 1
        if authority_count != expected_authority_count:
            raise RuntimeError(
                f"Code-authority pin count in {name}={authority_count}, expected {expected_authority_count}"
            )
        if text.count(AUTHORITY_SCHEMA_MARKER) != expected_authority_count:
            raise RuntimeError(f"Code-authority schema marker count is invalid in {name}")
        if name != "00_colab_bootstrap.ipynb" and "_AUTHORITY_BOOTSTRAP_ALLOWED = True" in text:
            raise RuntimeError(f"Downstream notebook may not establish or rotate authority: {name}")
    for name in (
        "03_calibration_and_ci.ipynb",
        "05_hrv_domain_and_robustness.ipynb",
        "07_results_freeze.ipynb",
    ):
        text = "\n".join(source(cell) for cell in load(name)["cells"])
        if AUTHENTICATED_BOOTSTRAP_UNIT not in text or GROUP_SEMANTICS not in text:
            raise RuntimeError(f"Authenticated bootstrap contract missing from {name}")
        if "freeze_group_sidecar_v1" not in text:
            raise RuntimeError(f"Freeze group-sidecar binding check missing from {name}")
        for stale in (
            "chapman_record_subject",
            "chapman_record_one_record_per_subject",
            "one_chapman_record_per_subject",
        ):
            if stale in text:
                raise RuntimeError(f"Stale bootstrap contract {stale!r} remains in {name}")
    final_text = "\n".join(source(cell) for cell in load("07_results_freeze.ipynb")["cells"])
    for token in (
        "--strict --full-sha",
        "47_forensic_notebook_audit.py",
        "notebook_forensic_audit.md",
        "table_paired_inference_audit.csv",
        "strict_full_sha_authority_update_v3",
        "--source-conflict-policy newer",
        "final_pipeline_storage_audit_post_publish_strict_full_sha.log",
    ):
        if token not in final_text:
            raise RuntimeError(f"Notebook 07 final gate token missing: {token}")
    notebook07 = load("07_results_freeze.ipynb")
    gate_index = next(
        index for index, cell in enumerate(notebook07["cells"])
        if "FORENSIC_NOTEBOOK07_FINAL_GATE" in source(cell)
    )
    export_index = next(
        index for index, cell in enumerate(notebook07["cells"])
        if "FINAL_TABLE_EXPORT_DIR" in source(cell)
    )
    if export_index <= gate_index:
        raise RuntimeError("Notebook 07 convenience export must run after strict forensic/storage gates")
    for index, cell in enumerate(notebook07["cells"][:gate_index]):
        text = source(cell)
        if "final_evidence_tables" in text and "shutil.copy2" in text:
            raise RuntimeError(f"Notebook 07 cell {index} exports final evidence before strict gates")

    notebook02a_text = "\n".join(
        source(cell) for cell in load("02a_retrain_best_ema.ipynb")["cells"]
    )
    for token in (
        "stage_run_id_durable_stream_v1",
        "'history' / 'retrain_best_ema_train'",
        "run(\n        training_command",
        BASE_INSTALLER_MARKER,
        BASE_INSTALLER_SCHEMA_MARKER,
        MAMBA_MARKER,
        MAMBA_SCHEMA_MARKER,
        "expected exactly one",
    ):
        if token not in notebook02a_text:
            raise RuntimeError(f"Notebook 02a direct-run contract token missing: {token}")
    notebook02a = load("02a_retrain_best_ema.ipynb")
    training_cells = [
        source(cell)
        for cell in notebook02a["cells"]
        if "FORENSIC_RETRAIN_STREAMING_LOG_CAPABILITY" in source(cell)
    ]
    if len(training_cells) != 1 or "subprocess.Popen" in training_cells[0]:
        raise RuntimeError("Notebook 02a training cell bypasses the forensic run-history wrapper")


def main() -> None:
    integrate_notebook02()
    integrate_notebook00()
    integrate_notebook01()
    integrate_installer_consumers()
    integrate_notebook02a_training_log()
    integrate_remaining_run_history()
    integrate_notebook03_strict_inputs()
    normalize_bootstrap_contracts()
    integrate_code_authority()
    integrate_notebook07_final_gate()
    validate()
    print("Forensic notebook integration complete and validated.")


if __name__ == "__main__":
    main()
