# TheSeriusProject: configure and build from repo root.
# Usage: ./.cursor/skills/theseriusproject/scripts/build.ps1 [Release|Debug]
$Config = if ($args[0]) { $args[0] } else { "Release" }
$Root = Split-Path (Split-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) -Parent) -Parent
$BuildDir = Join-Path $Root "build"

if (-not (Test-Path $BuildDir)) {
    New-Item -ItemType Directory -Path $BuildDir | Out-Null
}
Push-Location $BuildDir
try {
    if (-not (Test-Path "CMakeCache.txt")) {
        Write-Host "Configuring CMake..."
        & cmake $Root -DCMAKE_BUILD_TYPE=$Config
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    Write-Host "Building ($Config)..."
    & cmake --build . --config $Config
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
