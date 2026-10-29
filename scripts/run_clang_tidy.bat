@echo off
setlocal
set CLANG_TIDY=C:\BuildTools\VC\Tools\Llvm\x64\bin\clang-tidy.exe
set BUILD_DIR=%~dp0..\build-clangd

if not exist "%CLANG_TIDY%" (
  echo clang-tidy not found: %CLANG_TIDY%
  exit /b 1
)

echo Running clang-tidy on src/memory/memory_pool.cpp
"%CLANG_TIDY%" -p "%BUILD_DIR%" "%~dp0..\src\memory\memory_pool.cpp" --checks="-*,readability-*,bugprone-*,performance-*,modernize-*,-modernize-use-trailing-return-type"
