# soak_nightly.ps1 - launch a supervised soak stress test.
#
# Starts `serius serve` (with dedicated ops port for /health + /metrics),
# waits until the model is loaded, runs `serius stress -t soak` against it,
# then stops the server. Safe to run unattended (Task Scheduler).
#
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\soak_nightly.ps1
#         [-DurationSec 14400] [-Users 4] [-MaxTokens 64] [-Tag nightly]
# Logs:   stress_results\soak_server.log / soak_server.err.log
#         stress_results\soak_build.log   (rebuild output, best effort)
# Report: stress_results\soak_<Tag>.json

param(
    [int]$DurationSec = 14400,   # 4h default
    [int]$Users = 4,
    [int]$MaxTokens = 64,
    [string]$Tag = 'nightly'
)

$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$serius = 'build-ninja\src\serius.exe'
$model  = 'models\smollm2-q4\SmolLM2-135M-Instruct-Q4_0.gguf'
$port   = 8080
$opsPort = 8081
$durationSec = $DurationSec
$maxTokens = $MaxTokens

New-Item -ItemType Directory -Force 'stress_results' | Out-Null

# --- 1) Best-effort rebuild so the soak tests the latest code -----------
$vcvars = 'C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
if (Test-Path $vcvars) {
    cmd /c "call `"$vcvars`" >nul && ninja -C build-ninja src/serius.exe" `
        *> 'stress_results\soak_build.log'
    if ($LASTEXITCODE -ne 0) {
        Add-Content 'stress_results\soak_build.log' "`n[soak] WARNING: rebuild failed (exit $LASTEXITCODE) - using existing binary"
    }
}

if (-not (Test-Path $serius)) {
    Write-Error "serius.exe not found at $serius"
    exit 1
}

# --- 2) Kill stale server/stress processes from previous runs ----------
# Match any binary named serius* (covers build-ninja/src/serius.exe and
# ad-hoc copies like stress_results/serius_srv.exe) so port 8080/8081 are free.
Get-Process -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -like 'serius*' -and $_.Path } |
    Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# --- 3) Start the server -------------------------------------------------
$serverLog  = 'stress_results\soak_server.log'
$serverErr  = 'stress_results\soak_server.err.log'
Remove-Item $serverLog, $serverErr -ErrorAction SilentlyContinue

$server = Start-Process -FilePath $serius `
    -ArgumentList @('serve', '--model', $model, '--port', "$port",
                    '--health-port', "$opsPort") `
    -RedirectStandardOutput $serverLog `
    -RedirectStandardError  $serverErr `
    -PassThru -WindowStyle Hidden

Write-Output "[soak] server pid=$($server.Id), waiting for model load..."

$ready = $false
for ($i = 0; $i -lt 180; $i++) {
    try {
        $h = Invoke-RestMethod -Uri "http://localhost:$opsPort/health" -TimeoutSec 3
        if ($h.model_loaded) { $ready = $true; break }
    } catch { }
    if ($server.HasExited) { break }
    Start-Sleep -Seconds 1
}

if (-not $ready) {
    Write-Error "[soak] server did not become ready (see $serverLog)"
    if (-not $server.HasExited) { Stop-Process -Id $server.Id -Force }
    exit 1
}
Write-Output "[soak] model loaded - starting stress run (${durationSec}s, $Users users)"

# --- 4) Run the soak ------------------------------------------------------
$outJson = "stress_results\soak_$Tag.json"
$outLog  = "stress_results\soak_$Tag.log"
& $serius stress -t soak `
    -s "http://localhost:$port" `
    --health-url "http://localhost:$opsPort/health" `
    -u $Users -d $durationSec -n $maxTokens `
    -o $outJson -f json `
    *> $outLog
$rc = $LASTEXITCODE
Write-Output "[soak] stress finished rc=$rc"

# --- 5) Stop the server ---------------------------------------------------
if (-not $server.HasExited) { Stop-Process -Id $server.Id -Force }
Write-Output "SOAK_DONE rc=$rc out=$outJson"
exit $rc
