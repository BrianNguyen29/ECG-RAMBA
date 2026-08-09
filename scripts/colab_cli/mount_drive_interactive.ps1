[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Session,
    [ValidateSet("oauth2", "adc")]
    [string]$Auth = "oauth2",
    [string]$Distro = "Ubuntu",
    [ValidateRange(0, 600)]
    [int]$AutoConfirmAfterSeconds = 0,
    [ValidateRange(1, 120)]
    [int]$VerifyAttempts = 60,
    [ValidateRange(1, 30)]
    [int]$VerifyDelaySeconds = 2
)

$ErrorActionPreference = "Stop"
$ColabExecutable = "/home/uong_guyen/.local/bin/colab"

if ($Session -notmatch "^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$") {
    throw "Invalid Colab session name: $Session"
}
if ($Distro -notmatch "^[A-Za-z0-9._-]+$") {
    throw "Invalid WSL distribution name: $Distro"
}

function Quote-Bash([string]$Value) {
    return "'" + $Value.Replace("'", "'""'""'") + "'"
}

# Colab CLI 0.6.0 reads the Drive-consent acknowledgement directly from
# /dev/tty. `script` supplies a pseudo-terminal and forwards PowerShell input.
$ColabCommand = (
    "{0} --auth={1} drivemount -s {2} /content/drive" -f
    $ColabExecutable,
    $Auth,
    $Session
)
$PtyCommand = "exec script -qefc $(Quote-Bash $ColabCommand) /dev/null"

$ProcessInfo = [System.Diagnostics.ProcessStartInfo]::new()
$ProcessInfo.FileName = "wsl.exe"
$HasArgumentList = (
    $null -ne $ProcessInfo.PSObject.Properties['ArgumentList'] -and
    $null -ne $ProcessInfo.ArgumentList
)
if ($HasArgumentList) {
    # PowerShell 7 / modern .NET: preserve each argument without shell re-parsing.
    $ProcessInfo.ArgumentList.Add("-d")
    $ProcessInfo.ArgumentList.Add($Distro)
    $ProcessInfo.ArgumentList.Add("--exec")
    $ProcessInfo.ArgumentList.Add("bash")
    $ProcessInfo.ArgumentList.Add("-lc")
    $ProcessInfo.ArgumentList.Add($PtyCommand)
} else {
    # Windows PowerShell 5.1 exposes ArgumentList as null. The PTY command uses
    # single-quoted Bash content, so only embedded double quotes need escaping
    # for the Windows process command line.
    $EscapedPtyCommand = $PtyCommand.Replace('"', '\"')
    $ProcessInfo.Arguments = "-d $Distro --exec bash -lc `"$EscapedPtyCommand`""
}
$ProcessInfo.UseShellExecute = $false
$ProcessInfo.RedirectStandardOutput = $true
$ProcessInfo.RedirectStandardInput = $true
$ProcessInfo.RedirectStandardError = $false

$Process = [System.Diagnostics.Process]::new()
$Process.StartInfo = $ProcessInfo
try {
    $Started = $Process.Start()
} catch {
    throw "Could not start wsl.exe for the Colab Drive mount: $($_.Exception.Message)"
}
if (-not $Started) {
    throw "wsl.exe did not start the Colab Drive mount process."
}
$OutputReader = $Process.StandardOutput
$InputWriter = $Process.StandardInput
if ($null -eq $OutputReader -or $null -eq $InputWriter) {
    $Process.WaitForExit()
    throw (
        "Colab Drive mount process did not expose redirected input/output streams. " +
        "exit_code=$($Process.ExitCode). Verify wsl.exe, the selected distro, and the Colab CLI installation."
    )
}

$ObservedMountError = $false
$ConsentRequested = $false
$ConsentUrlPattern = "https://(?:accounts\\.google\\.com|colab\\.research\\.google\\.com)/\\S+"
while (($Line = $OutputReader.ReadLine()) -ne $null) {
    Write-Host $Line
    if (
        $Line -match "\[colab\] Error propagating:" -or
        $Line -match "ValueError.*mount failed"
    ) {
        $ObservedMountError = $true
    }
    if (-not $ConsentRequested -and $Line -match $ConsentUrlPattern) {
        $ConsentRequested = $true
        $ConsentUrl = $Matches[0]
        Start-Process $ConsentUrl
        if ($AutoConfirmAfterSeconds -gt 0) {
            Write-Host (
                "Approve Google Drive access in the browser. " +
                "Continuing automatically in $AutoConfirmAfterSeconds seconds..."
            )
            Start-Sleep -Seconds $AutoConfirmAfterSeconds
        } else {
            [void](Read-Host "Approve Google Drive access in the browser, then press Enter here")
        }
        $InputWriter.WriteLine()
        $InputWriter.Flush()
    }
}

$Process.WaitForExit()
$VerifyPath = "/content/drive/MyDrive/ECG-Ramba"
$VerifyOutput = @()
$VerifyExitCode = 1
for ($Attempt = 1; $Attempt -le $VerifyAttempts; $Attempt++) {
    $VerifyOutput = & wsl.exe -d $Distro --exec $ColabExecutable `
        "--auth=$Auth" ls -s $Session $VerifyPath 2>&1
    $VerifyExitCode = $LASTEXITCODE
    if ($VerifyExitCode -eq 0) {
        break
    }
    if ($Attempt -lt $VerifyAttempts) {
        if ($Attempt -eq 1 -or $Attempt % 10 -eq 0) {
            Write-Host (
                "Drive is mounted but the ECG-Ramba project root is not visible yet " +
                "(attempt $Attempt/$VerifyAttempts). Waiting for DriveFS..."
            )
        }
        Start-Sleep -Seconds $VerifyDelaySeconds
    }
}
if (
    $Process.ExitCode -ne 0 -or
    $ObservedMountError -or
    $VerifyExitCode -ne 0
) {
    if ($VerifyOutput) {
        Write-Host ($VerifyOutput | Out-String)
    }
    throw (
        "Colab Drive mount verification failed. " +
        "cli_exit=$($Process.ExitCode) observed_mount_error=$ObservedMountError " +
        "verify_exit=$VerifyExitCode verify_path=$VerifyPath"
    )
}
Write-Host "Google Drive and ECG-Ramba project root are readable for Colab session: $Session"
