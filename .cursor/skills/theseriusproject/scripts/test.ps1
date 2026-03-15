# TheSeriusProject: run tests from repo root.
# Usage: ./.cursor/skills/theseriusproject/scripts/test.ps1
$Root = Split-Path (Split-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) -Parent) -Parent
$BuildDir = Join-Path $Root "build"
if (-not (Test-Path $BuildDir)) {
    Write-Error "Run build first. build/ not found."
    exit 1
}
Push-Location $BuildDir
try {
    & ctest --output-on-failure
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
