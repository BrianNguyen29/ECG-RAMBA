# ECG-RAMBA Colab CLI Pipeline

## Scope

This pipeline executes the reviewed Notebook 00-07 workflow through the
official Google Colab CLI while retaining the notebooks as the executable
provenance artifacts. It does not use Colab MCP.

The source of truth is:

- stage manifest: `configs/colab_cli_pipeline.json`
- stage builder: `scripts/colab_cli/stage_notebook.py`
- session launcher: `scripts/colab_cli/pipeline.py`
- Windows/WSL bridge: `scripts/colab_cli/run_pipeline.ps1`

The canonical artifact root remains:

`/content/drive/MyDrive/ECG-Ramba/revision_artifacts/reports/revision`

## Safety properties

- A100 stages assert that the assigned GPU name contains `A100`.
- Notebook 02 CPU feature extraction and A100 inference are separate stages.
- Notebook 02a retraining is disabled by default.
- The current immutable code-authority commit/ref is injected into every stage.
- Before building, every source notebook is checked against the annotated
  authority tag; the generated stage is parsed directly from that immutable
  Git blob rather than mutable working-tree bytes.
- Each generated stage records SHA256 for the pipeline manifest, stage builder,
  launcher, and immutable source-notebook blob.
- Generated stage notebooks contain the source notebook SHA256.
- A stage prints a completion marker only after all selected cells finish.
- A zero CLI exit code is accepted only when that exact completion marker is
  present in the streamed log.
- Failed or interrupted sessions remain active for inspection and resume.
- Successful sessions are stopped unless `--keep` is requested.
- Every run streams to a local text log, preserves the executed
  `<stage>_output.ipynb`, and exports Colab session history to `.ipynb`.

## One-time Windows setup

Open PowerShell in the repository:

```powershell
.\scripts\colab_cli\run_pipeline.ps1 -Action install
```

The launcher uses WSL2 because Google Colab CLI does not support native
Windows. Verify the plan:

```powershell
.\scripts\colab_cli\run_pipeline.ps1 -Action validate
.\scripts\colab_cli\run_pipeline.ps1 -Action plan
```

Configure ADC once. This is interactive and requires a browser:

```powershell
.\scripts\colab_cli\run_pipeline.ps1 -Action auth-setup
.\scripts\colab_cli\run_pipeline.ps1 -Action auth-check
```

The setup requests the four scopes required by the Colab assignment and
keep-alive services. `auth-check` fails before allocating a VM if any scope is
missing. OAuth2 remains available with `-Auth oauth2`, but ADC is the default
for this automation.

Drive mount is a separate VM-side consent flow. Each newly provisioned session
will pause at `colab drivemount`; complete the browser consent and return to the
PowerShell terminal. This cannot be made unattended by Colab CLI.

## Running one stage

```powershell
.\scripts\colab_cli\run_pipeline.ps1 `
  -Action run `
  -Stage nb02_features_cpu
```

The launcher provisions a named runtime, asks for Drive consent, executes the
generated notebook, exports the log, and stops the runtime after success.

For an A100 stage:

```powershell
.\scripts\colab_cli\run_pipeline.ps1 `
  -Action run `
  -Stage nb02_a100
```

If A100 allocation is unavailable, the stage fails. It never falls back to a
different GPU.

## Full sequence

Run all enabled stages:

```powershell
.\scripts\colab_cli\run_pipeline.ps1 -Action run-all
```

The enabled sequence is:

1. `nb00_cpu`
2. `nb01_cpu`
3. `nb02_features_cpu`
4. `nb02_a100`
5. `nb03_cpu`
6. `nb04_a100`
7. `nb05_predictions_a100`
8. `nb05_finalize_cpu`
9. `nb06_a100`
10. `nb07_cpu`

`run-all` is sequential but not completely unattended because each new
CPU/A100 session requires Drive consent. For expensive runs, stage-by-stage
execution is preferable: it makes the CPU-to-A100 handoff and the canonical
mirror publish checkpoint explicit.

Notebook 02a is intentionally skipped. Run it only after documented protocol
review:

```powershell
.\scripts\colab_cli\run_pipeline.ps1 `
  -Action run `
  -Stage nb02a_retrain_a100 `
  -IncludeDisabled
```

## Resume and diagnostics

On failure, the runtime remains active:

```powershell
.\scripts\colab_cli\run_pipeline.ps1 -Action sessions
.\scripts\colab_cli\run_pipeline.ps1 -Action run -Stage <stage-id>
```

The second command reuses the stable stage session name. Add `-NoMount` when
Drive is already mounted, or `-Remount` to explicitly repeat the mount flow.

Keep a successful runtime for manual inspection:

```powershell
.\scripts\colab_cli\run_pipeline.ps1 `
  -Action run `
  -Stage nb04_a100 `
  -Keep
```

Local logs are written under:

`reports/revision/logs/colab_cli/<stage-id>/`

The canonical notebook and runner logs continue to be published to Drive by
the notebook cells themselves.

## Dry run

Build and inspect commands without allocating resources:

```powershell
.\scripts\colab_cli\run_pipeline.ps1 `
  -Action run-all `
  -DryRun
```

Generated notebooks are stored under:

`~/.cache/ecg-ramba-colab-cli/stages/`

They are derived artifacts and are not committed.
