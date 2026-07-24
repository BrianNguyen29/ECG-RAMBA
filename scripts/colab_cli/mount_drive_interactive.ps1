[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Session,
    [ValidateSet("oauth2", "adc")]
    [string]$Auth = "oauth2",
    [string]$Distro = "Ubuntu",
    [ValidateRange(0, 600)]
    [int]$AutoConfirmAfterSeconds = 0
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
$ProcessInfo.ArgumentList.Add("-d")
$ProcessInfo.ArgumentList.Add($Distro)
$ProcessInfo.ArgumentList.Add("--exec")
$ProcessInfo.ArgumentList.Add("bash")
$ProcessInfo.ArgumentList.Add("-lc")
$ProcessInfo.ArgumentList.Add($PtyCommand)
$ProcessInfo.UseShellExecute = $false
$ProcessInfo.RedirectStandardOutput = $true
$ProcessInfo.RedirectStandardInput = $true
$ProcessInfo.RedirectStandardError = $false

$Process = [System.Diagnostics.Process]::new()
$Process.StartInfo = $ProcessInfo
[void]$Process.Start()

while (($Line = $Process.StandardOutput.ReadLine()) -ne $null) {
    Write-Host $Line
    if ($Line -match "^https://accounts\.google\.com/") {
        Start-Process $Line
        if ($AutoConfirmAfterSeconds -gt 0) {
            Write-Host (
                "Approve Google Drive access in the browser. " +
                "Continuing automatically in $AutoConfirmAfterSeconds seconds..."
            )
            Start-Sleep -Seconds $AutoConfirmAfterSeconds
        } else {
            [void](Read-Host "Approve Google Drive access in the browser, then press Enter here")
        }
        $Process.StandardInput.WriteLine()
        $Process.StandardInput.Flush()
    }
}

$Process.WaitForExit()
if ($Process.ExitCode -ne 0) {
    throw "Colab Drive mount failed with exit code $($Process.ExitCode)."
}
Write-Host "Google Drive mounted for Colab session: $Session"
