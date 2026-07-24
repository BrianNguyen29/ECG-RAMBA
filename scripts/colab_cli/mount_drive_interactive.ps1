[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Session,
    [ValidateSet("oauth2", "adc")]
    [string]$Auth = "oauth2",
    [string]$Distro = "Ubuntu"
)

$ErrorActionPreference = "Stop"
$ColabExecutable = "/home/uong_guyen/.local/bin/colab"
$ProcessInfo = [System.Diagnostics.ProcessStartInfo]::new()
$ProcessInfo.FileName = "wsl.exe"
$ProcessInfo.Arguments = (
    "-d {0} --exec {1} --auth={2} drivemount -s {3} /content/drive" -f
    $Distro,
    $ColabExecutable,
    $Auth,
    $Session
)
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
        [void](Read-Host "Approve Google Drive access in the browser, then press Enter here")
        $Process.StandardInput.WriteLine()
        $Process.StandardInput.Flush()
    }
}

$Process.WaitForExit()
if ($Process.ExitCode -ne 0) {
    throw "Colab Drive mount failed with exit code $($Process.ExitCode)."
}
Write-Host "Google Drive mounted for Colab session: $Session"
