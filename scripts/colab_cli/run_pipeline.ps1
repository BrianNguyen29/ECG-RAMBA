[CmdletBinding()]
param(
    [ValidateSet("install", "auth-setup", "plan", "validate", "auth-check", "sessions", "build", "run", "run-all")]
    [string]$Action = "plan",
    [string]$Stage,
    [string]$FromStage,
    [string]$ToStage,
    [ValidateSet("oauth2", "adc")]
    [string]$Auth = "oauth2",
    [string]$Distro = "Ubuntu",
    [switch]$Keep,
    [switch]$NoMount,
    [switch]$Remount,
    [switch]$IncludeDisabled,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$DriveLetter = $RepoRoot.Substring(0, 1).ToLowerInvariant()
$Relative = $RepoRoot.Substring(2).Replace("\", "/")
$WslRepoRoot = "/mnt/$DriveLetter$Relative"

function Quote-Bash([string]$Value) {
    return "'" + $Value.Replace("'", "'""'""'") + "'"
}

if ($Action -eq "install") {
    $Command = "cd $(Quote-Bash $WslRepoRoot) && bash scripts/colab_cli/install_wsl.sh"
    & wsl.exe -d $Distro -- bash -lc $Command
    exit $LASTEXITCODE
}

if ($Action -eq "auth-setup") {
    if ($Auth -eq "oauth2") {
        $SetupScript = "scripts/colab_cli/setup_oauth2.sh"
    } else {
        $SetupScript = "scripts/colab_cli/setup_adc.sh"
    }
    $Command = "cd $(Quote-Bash $WslRepoRoot) && bash $(Quote-Bash $SetupScript)"
    & wsl.exe -d $Distro -- bash -lc $Command
    exit $LASTEXITCODE
}

$Arguments = @(
    "python3",
    "scripts/colab_cli/pipeline.py",
    "--auth",
    $Auth,
    $Action
)

if ($Stage) { $Arguments += @("--stage", $Stage) }
if ($FromStage) { $Arguments += @("--from-stage", $FromStage) }
if ($ToStage) { $Arguments += @("--to-stage", $ToStage) }
if ($Keep) { $Arguments += "--keep" }
if ($NoMount) { $Arguments += "--no-mount" }
if ($Remount) { $Arguments += "--remount" }
if ($IncludeDisabled) { $Arguments += "--include-disabled" }
if ($DryRun) { $Arguments += "--dry-run" }

$BashArguments = ($Arguments | ForEach-Object { Quote-Bash $_ }) -join " "
$Command = @"
export PATH="`$HOME/.local/bin:`$HOME/.cargo/bin:`$PATH"
cd $(Quote-Bash $WslRepoRoot)
$BashArguments
"@

& wsl.exe -d $Distro -- bash -lc $Command
exit $LASTEXITCODE
