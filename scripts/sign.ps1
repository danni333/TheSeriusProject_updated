#Requires -Version 5.1
<#
.SYNOPSIS
    Code-sign Serius binaries/installer with signtool.

.DESCRIPTION
    Signs every .exe/.dll/.msi under -Path (recursively with -Recurse).
    The certificate can come from:
      - the Windows certificate store, selected by -Thumbprint or -CertName
      - a PFX file via -Pfx/-Password (or env SERIUS_SIGN_PFX / SERIUS_SIGN_PASSWORD)

    signtool.exe is located automatically under the Windows SDK
    (C:\Program Files (x86)\Windows Kits\10\bin\<ver>\x64\signtool.exe).

    Notes:
      - For public distribution use a standard (OV) or EV code-signing cert;
        SmartScreen reputation builds over time with a regular cert.
      - Timestamping (-t http://timestamp.digicert.com) keeps signatures valid
        after cert expiry — enabled by default, disable with -NoTimestamp.

.PARAMETER Path
    File or directory to sign.

.EXAMPLE
    .\scripts\sign.ps1 -Path dist\stage -Recurse -Thumbprint "A1B2C3..."
    .\scripts\sign.ps1 -Path Serius-Setup-0.1.0-x64.msi -Pfx .\cert.pfx -Password "..."
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Path,

    [switch]$Recurse,

    [string]$Thumbprint,
    [string]$CertName,
    [string]$Pfx,
    [string]$Password,

    [string]$TimestampUrl = "http://timestamp.digicert.com",
    [switch]$NoTimestamp
)

$ErrorActionPreference = "Stop"

# ------------------------------------------------------------- find signtool ---
$signtool = Get-ChildItem "C:\Program Files (x86)\Windows Kits\10\bin\*\x64\signtool.exe" -ErrorAction SilentlyContinue |
            Sort-Object FullName -Descending | Select-Object -First 1 -ExpandProperty FullName
if (-not $signtool) { throw "signtool.exe not found - install the Windows SDK." }
Write-Host "signtool: $signtool"

# ------------------------------------------------------------- resolve cert ---
$SignCert = $null
$PfxPath  = $null

if ($Thumbprint) {
    $SignCert = Get-ChildItem Cert:\CurrentUser\My, Cert:\LocalMachine\My |
                Where-Object { $_.Thumbprint -eq $Thumbprint.Replace(' ', '') } | Select-Object -First 1
    if (-not $SignCert) { throw "Certificate with thumbprint $Thumbprint not found in CurrentUser\My or LocalMachine\My" }
} elseif ($CertName) {
    $SignCert = Get-ChildItem Cert:\CurrentUser\My, Cert:\LocalMachine\My |
                Where-Object { $_.Subject -like "*$CertName*" } | Select-Object -First 1
    if (-not $SignCert) { throw "No certificate matching '$CertName' in the store" }
} elseif ($Pfx) {
    $PfxPath = (Resolve-Path $Pfx).Path
    if (-not $Password) { $Password = $env:SERIUS_SIGN_PASSWORD }
} elseif ($env:SERIUS_SIGN_THUMBPRINT) {
    $Thumbprint = $env:SERIUS_SIGN_THUMBPRINT
    $SignCert = Get-ChildItem Cert:\CurrentUser\My, Cert:\LocalMachine\My |
                Where-Object { $_.Thumbprint -eq $Thumbprint.Replace(' ', '') } | Select-Object -First 1
    if (-not $SignCert) { throw "Certificate with thumbprint $Thumbprint not found" }
} elseif ($env:SERIUS_SIGN_PFX) {
    $PfxPath = $env:SERIUS_SIGN_PFX
    if (-not $Password) { $Password = $env:SERIUS_SIGN_PASSWORD }
} else {
    throw "No certificate specified. Use -Thumbprint/-CertName/-Pfx or SERIUS_SIGN_* env vars."
}

# --------------------------------------------------------------- collect files ---
$targets = @()
if (Test-Path $Path -PathType Container) {
    foreach ($e in @('*.exe', '*.dll', '*.msi')) {
        $targets += Get-ChildItem $Path -Recurse:$Recurse.IsPresent -Filter $e -File -ErrorAction SilentlyContinue
    }
} else {
    $targets += Get-Item $Path
}

if ($targets.Count -eq 0) { Write-Warning "No signable files (.exe/.dll/.msi) under $Path"; return }

# -------------------------------------------------------------------- sign ---
$timestamp = if ($NoTimestamp) { @() } else { @('/tr', $TimestampUrl, '/td', 'SHA256') }
$failed = @()

foreach ($f in $targets) {
    $args = @('sign', '/fd', 'SHA256', '/v') + $timestamp
    if ($PfxPath) {
        $args += @('/f', $PfxPath)
        if ($Password) { $args += @('/p', $Password) }
    } else {
        $args += @('/sha1', $SignCert.Thumbprint)
    }
    $args += $f.FullName

    Write-Host "Signing: $($f.Name)"
    & $signtool @args
    if ($LASTEXITCODE -ne 0) { $failed += $f.FullName }
}

if ($failed.Count -gt 0) {
    throw "Signing failed for $($failed.Count) file(s):`n$($failed -join "`n")"
}
Write-Host "All files signed."
