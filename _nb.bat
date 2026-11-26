@echo off
call "C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
ninja -C build-ninja %*
