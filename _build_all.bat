@echo off
call "C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d "C:\Users\ddech\OneDrive\Desktop\sgravitamelotutto\TheSeriusProject_updated"
ninja -C build-ninja
