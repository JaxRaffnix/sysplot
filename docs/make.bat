@ECHO OFF
pushd %~dp0

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)

set SOURCEDIR=.
set BUILDDIR=_build

if "%1" == "" goto help

REM ---- CUSTOM CLEAN ----
if "%1" == "clean" (
    %SPHINXBUILD% -M clean %SOURCEDIR% %BUILDDIR%
	echo Removing '_autosummary'...
    if exist _autosummary rmdir /s /q _autosummary
	echo Removing '_auto_examples'...
    if exist _auto_examples rmdir /s /q _auto_examples
    goto end
)

REM ---- NORMAL SPHINX TARGET ----
%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%

:end
popd