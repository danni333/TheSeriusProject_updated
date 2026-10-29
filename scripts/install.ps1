# Serius One-Line Installer for Windows

$ErrorActionPreference = "Stop"

$Version = if ($env:VERSION) { $env:VERSION } else { "latest" }
$InstallDir = if ($env:INSTALL_DIR) { $env:INSTALL_DIR } else { "$env:USERPROFILE\.serius\bin" }
$Repo = "your-org/serius"

Write-Host "Installing Serius LLM Engine..."

# Detect architecture
$Arch = if ([System.Environment]::Is64BitOperatingSystem) { "x64" } else { "x86" }
$File = "serius-windows-$Arch.zip"

# Download URL
if ($Version -eq "latest") {
    $DownloadUrl = "https://github.com/${Repo}/releases/latest/download/${File}"
} else {
    $DownloadUrl = "https://github.com/${Repo}/releases/download/${Version}/${File}"
}

Write-Host "Downloading from: $DownloadUrl"

# Create install directory
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null

# Download
$TempFile = "$env:TEMP\serius.zip"
Invoke-WebRequest -Uri $DownloadUrl -OutFile $TempFile

# Extract
Expand-Archive -Path $TempFile -DestinationPath $InstallDir -Force

# Add to PATH if not already present
$Path = [Environment]::GetEnvironmentVariable("Path", "User")
if ($Path -notlike "*$InstallDir*") {
    [Environment]::SetEnvironmentVariable("Path", "$Path;$InstallDir", "User")
    Write-Host "Added $InstallDir to PATH. Please restart your terminal."
}

# Cleanup
Remove-Item $TempFile

# Verify
$SeriusPath = Join-Path $InstallDir "serius.exe"
if (Test-Path $SeriusPath) {
    Write-Host "Serius installed successfully!"
    & $SeriusPath --version
} else {
    Write-Host "Installation failed"
    exit 1
}
