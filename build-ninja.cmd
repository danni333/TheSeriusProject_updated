@echo off
rem Build TheSeriusProject with Ninja from any shell (git-bash included).
rem Usage: build-ninja.cmd [target ...]   (no args = full build)
call "C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (
    echo vcvars64.bat failed - check VS Build Tools installation
    exit /b 1
)
ninja -C "%~dp0build-ninja" %*
