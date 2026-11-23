#Requires -Version 5.1
<#
.SYNOPSIS
    Build the Serius Windows distributables: portable ZIP and MSI installer.

.DESCRIPTION
    Stages the release binary plus its runtime dependencies, then produces:
      - serius-windows-x64.zip            (portable package, matches scripts/install.ps1)
      - Serius-Setup-<ver>-x64.msi        (WiX installer, if `wix` is on PATH)
      - *.sha256 checksums for each artifact

    Runtime DLLs bundled next to serius.exe:
      - VC++ runtime (vcruntime140*, msvcp140*, vcomp140*) from the local MSVC redist
      - CUDA cublas/cublasLt only with -WithCuda (adds ~550 MB)

    Code signing is optional: pass -SignThumbprint/-SignPfx or set
    SERIUS_SIGN_PFX / SERIUS_SIGN_PASSWORD / SERIUS_SIGN_THUMBPRINT.
    See scripts/sign.ps1 for details.

.PARAMETER BuildDir
    CMake build directory containing src\serius.exe. Default: build-lean
    (Release, CUDA off — the right choice for customer machines).

.PARAMETER WithCuda
    Also bundle cublas64_*.dll / cublasLt64_*.dll / cudart64_*.dll from -CudaBin.
    Only use with a CUDA-enabled build (e.g. -BuildDir build_cuda).

.PARAMETER Version
    Package version. Default: SERIUS_VERSION from the build's CMakeCache, else 0.1.0.

.EXAMPLE
    .\scripts\package.ps1
    .\scripts\package.ps1 -BuildDir build_cuda -WithCuda -SignThumbprint "A1B2..."
#>
[CmdletBinding()]
param(
    [string]$BuildDir = "build-lean",
    [switch]$WithCuda,
    [string]$CudaBin = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64",
    [string]$Version,
    [string]$OutDir = "dist",
    [string]$SignThumbprint,
    [string]$SignPfx,
    [string]$SignPassword,
    [string]$WiXDir = "",
    [switch]$SkipMsi
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

# ---------------------------------------------------------------- version ---
if (-not $Version) {
    $cache = Join-Path $BuildDir "CMakeCache.txt"
    if (Test-Path $cache) {
        $m = Select-String -Path $cache -Pattern '^SERIUS_VERSION:STRING=(.+)$' | Select-Object -First 1
        if ($m) { $Version = $m.Matches[0].Groups[1].Value.Trim('"') }
    }
}
if (-not $Version) { $Version = "0.1.0" }
$Version = $Version -replace '[^0-9A-Za-z.\-]', '-'

# ------------------------------------------------------------------ paths ---
$ExePath  = Join-Path $BuildDir "src\serius.exe"
if (-not (Test-Path $ExePath)) { throw "serius.exe not found at $ExePath - build first." }

$StageDir = Join-Path $OutDir "stage"
$ZipPath  = Join-Path $OutDir "serius-windows-x64.zip"
$MsiPath  = Join-Path $OutDir "Serius-Setup-$Version-x64.msi"

Remove-Item $StageDir -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $StageDir | Out-Null
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

# ------------------------------------------------------------- stage files ---
Copy-Item $ExePath (Join-Path $StageDir "serius.exe")

$UiDir = Join-Path $RepoRoot "ui"
if (Test-Path $UiDir) {
    Copy-Item $UiDir (Join-Path $StageDir "ui") -Recurse
} else {
    Write-Warning "ui/ directory not found - 'serius serve' web UI will be unavailable."
}

# VC++ runtime DLLs - newest MSVC redist under common roots
$VcDlls = @("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll", "msvcp140_1.dll", "vcomp140.dll", "concrt140.dll")
$VcRoots = @(
    "${env:ProgramFiles(x86)}\Microsoft Visual Studio",
    "C:\BuildTools\VC\Redist",
    "${env:ProgramFiles}\Microsoft Visual Studio"
)
foreach ($dll in $VcDlls) {
    $hit = Get-ChildItem $VcRoots -Recurse -Filter $dll -ErrorAction SilentlyContinue |
           Where-Object { $_.FullName -match '\\x64\\' } |
           Sort-Object FullName -Descending | Select-Object -First 1
    if ($hit) {
        Copy-Item $hit.FullName $StageDir
    } else {
        Write-Warning "VC runtime DLL not found: $dll (target PCs need the VC++ 2015-2022 redist)"
    }
}

# CUDA runtime DLLs (only with -WithCuda)
if ($WithCuda) {
    foreach ($dll in @("cublas64_13.dll", "cublasLt64_13.dll", "cudart64_13.dll")) {
        $src = Join-Path $CudaBin $dll
        if (Test-Path $src) {
            Copy-Item $src $StageDir
        } else {
            Write-Warning "CUDA DLL not found: $src"
        }
    }
}

# End-user quickstart
@"
Serius LLM Engine $Version (Windows x64 portable)
================================================

Run from this folder:

  .\serius.exe doctor                      # check CPU/GPU/model readiness
  .\serius.exe serve -m <model.gguf> -p 8080 --ui ui
  .\serius.exe chat -m <model.gguf>

The web UI is served at http://localhost:8080 after 'serius serve'.

Requirements: Windows 10/11 x64. GPU acceleration needs an NVIDIA GPU with
recent drivers; everything else runs on CPU.
"@ | Set-Content (Join-Path $StageDir "README.txt") -Encoding UTF8

# ------------------------------------------------------------------ signing ---
$SignThumb = if ($SignThumbprint) { $SignThumbprint } else { $env:SERIUS_SIGN_THUMBPRINT }
$SignPfxF  = if ($SignPfx)         { $SignPfx }         else { $env:SERIUS_SIGN_PFX }
$SignPwd   = if ($SignPassword)    { $SignPassword }    else { $env:SERIUS_SIGN_PASSWORD }
$DoSign    = [bool]$SignThumb -or [bool]$SignPfxF

if ($DoSign) {
    & (Join-Path $PSScriptRoot "sign.ps1") -Path $StageDir -Recurse `
        -Thumbprint $SignThumb -Pfx $SignPfxF -Password $SignPwd
}

# ---------------------------------------------------------------------- zip ---
if (Test-Path $ZipPath) { Remove-Item $ZipPath -Force }
Compress-Archive -Path (Join-Path $StageDir "*") -DestinationPath $ZipPath -CompressionLevel Optimal
$zipHash = (Get-FileHash $ZipPath -Algorithm SHA256).Hash.ToLower()
"$zipHash  $(Split-Path -Leaf $ZipPath)" | Set-Content "$ZipPath.sha256" -Encoding ASCII
Write-Host ("ZIP : {0} ({1:N1} MB)" -f $ZipPath, ((Get-Item $ZipPath).Length / 1MB))

# ---------------------------------------------------------------------- msi ---
if (-not $SkipMsi) {
    # Locate WiX: prefer `wix` CLI (v4/v5) on PATH, else candle/light (v3 binaries)
    $wixExe   = (Get-Command wix -ErrorAction SilentlyContinue).Source
    if (-not $WiXDir) {
        foreach ($cand in @($env:SERIUS_WIX_DIR, "$env:USERPROFILE\tools\wix314", "C:\tools\wix314")) {
            if ($cand -and (Test-Path (Join-Path $cand "candle.exe"))) { $WiXDir = $cand; break }
        }
    }
    $candle = if ($WiXDir) { Join-Path $WiXDir "candle.exe" } else { $null }
    $light  = if ($WiXDir) { Join-Path $WiXDir "light.exe" }  else { $null }

    if (-not $wixExe -and -not ($candle -and (Test-Path $candle))) {
        Write-Warning "WiX not found - skipping MSI. Options: 'dotnet tool install --global wix', or WiX 3.14 binaries in ~/tools/wix314 (or -WiXDir)."
    } else {
        $wxsPath = Join-Path $OutDir "serius.wxs"
        $sb = New-Object System.Text.StringBuilder

        [void]$sb.AppendLine('<?xml version="1.0" encoding="UTF-8"?>')
        [void]$sb.AppendLine('<Wix xmlns="http://schemas.microsoft.com/wix/2006/wi">')
        [void]$sb.AppendLine("  <Product Id=`"*`" Name=`"Serius`" Language=`"1033`" Version=`"$Version`" Manufacturer=`"Serius`" UpgradeCode=`"7D2E5C41-9A3B-4E6F-8C1D-2B5A9F0E3D7C`">")
        [void]$sb.AppendLine('    <Package InstallerVersion="500" Compressed="yes" InstallScope="perMachine" />')
        [void]$sb.AppendLine('    <MajorUpgrade Schedule="afterInstallInitialize" AllowDowngrades="no" DowngradeErrorMessage="A newer version of Serius is already installed." />')
        [void]$sb.AppendLine('    <Media Id="1" Cabinet="serius.cab" EmbedCab="yes" />')
        [void]$sb.AppendLine('    <Directory Id="TARGETDIR" Name="SourceDir">')
        [void]$sb.AppendLine('      <Directory Id="ProgramFiles64Folder">')
        [void]$sb.AppendLine('        <Directory Id="INSTALLFOLDER" Name="Serius">')

        # Emit directory tree + components recursively
        $script:CompRefs = New-Object System.Collections.Generic.List[string]
        function Write-StageTree {
            param([System.IO.DirectoryInfo]$Dir, [string]$DirId, [int]$Depth)
            $indent = ' ' * (10 + $Depth * 2)
            foreach ($sub in $Dir.GetDirectories()) {
                $relSub = $sub.FullName.Substring($StageDir.Length).TrimStart('\')
                $dirId = "DIR_" + ([System.BitConverter]::ToString(
                    [System.Security.Cryptography.SHA1]::Create().ComputeHash(
                        [System.Text.Encoding]::UTF8.GetBytes($relSub))).Replace('-','').Substring(0,12))
                [void]$script:Sb.AppendLine("$indent<Directory Id=`"$dirId`" Name=`"$($sub.Name)`">")
                Write-StageTree -Dir $sub -DirId $dirId -Depth ($Depth + 1)
                [void]$script:Sb.AppendLine("$indent</Directory>")
            }
            foreach ($file in $Dir.GetFiles()) {
                $fileId = "F_" + ($file.Name + $file.Length).GetHashCode().ToString('X8')
                $compId  = "C_" + $fileId
                $src     = $file.FullName.Replace('\', '/')
                [void]$script:Sb.AppendLine("$indent<Component Id=`"$compId`" Guid=`"*`">")
                [void]$script:Sb.AppendLine("$indent  <File Id=`"$fileId`" Name=`"$($file.Name)`" Source=`"$src`" KeyPath=`"yes`" />")
                [void]$script:Sb.AppendLine("$indent</Component>")
                $script:CompRefs.Add($compId) | Out-Null
            }
        }
        $script:Sb = $sb
        Write-StageTree -Dir (Get-Item $StageDir) -DirId "INSTALLFOLDER" -Depth 2

        # PATH component: registry keypath + append install dir to system PATH
        [void]$sb.AppendLine('          <Component Id="C_SeriusPath" Guid="*">')
        [void]$sb.AppendLine('            <RegistryValue Root="HKLM" Key="SOFTWARE\Serius" Name="InstallDir" Type="string" Value="[INSTALLFOLDER]" KeyPath="yes" />')
        [void]$sb.AppendLine('            <Environment Id="SeriusPath" Name="Path" Value="[INSTALLFOLDER]" Permanent="no" Part="last" Action="set" System="yes" />')
        [void]$sb.AppendLine('          </Component>')

        [void]$sb.AppendLine('        </Directory>')
        [void]$sb.AppendLine('      </Directory>')
        [void]$sb.AppendLine('    </Directory>')
        [void]$sb.AppendLine('    <Feature Id="Main" Title="Serius" Level="1">')
        foreach ($c in $script:CompRefs) {
            [void]$sb.AppendLine("      <ComponentRef Id=`"$c`" />")
        }
        [void]$sb.AppendLine('      <ComponentRef Id="C_SeriusPath" />')
        [void]$sb.AppendLine('    </Feature>')
        [void]$sb.AppendLine('  </Product>')
        [void]$sb.AppendLine('</Wix>')

        Set-Content $wxsPath $sb.ToString() -Encoding UTF8

        if (Test-Path $MsiPath) { Remove-Item $MsiPath -Force }
        if ($wixExe) {
            & $wixExe build -arch x64 -out $MsiPath $wxsPath
        } else {
            $wixObj = Join-Path $OutDir "serius.wixobj"
            $wxsAbs = (Resolve-Path $wxsPath).Path
            $wixObjAbs = Join-Path (Get-Location) $wixObj
            & $candle -arch x64 -nologo -out $wixObjAbs $wxsAbs
            if ($LASTEXITCODE -ne 0) { throw "candle failed ($LASTEXITCODE)" }
            & $light -nologo -out $MsiPath $wixObjAbs -cultures:1033
            if ($LASTEXITCODE -ne 0) { throw "light failed ($LASTEXITCODE)" }
        }
        if ($LASTEXITCODE -ne 0) { throw "MSI build failed ($LASTEXITCODE)" }

        if ($DoSign) {
            & (Join-Path $PSScriptRoot "sign.ps1") -Path $MsiPath `
                -Thumbprint $SignThumb -Pfx $SignPfxF -Password $SignPwd
        }

        $msiHash = (Get-FileHash $MsiPath -Algorithm SHA256).Hash.ToLower()
        "$msiHash  $(Split-Path -Leaf $MsiPath)" | Set-Content "$MsiPath.sha256" -Encoding ASCII
        Write-Host ("MSI : {0} ({1:N1} MB)" -f $MsiPath, ((Get-Item $MsiPath).Length / 1MB))
    }
}

Write-Host ""
Write-Host "Done. Artifacts in $OutDir :"
Get-ChildItem $OutDir -File | ForEach-Object { Write-Host "  $($_.Name)" }
