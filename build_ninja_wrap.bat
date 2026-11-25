@echo off
call "C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
cmake --build build-ninja --target serius %*

