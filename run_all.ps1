param(
    [string]$PythonExe = ".\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"

& $PythonExe -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $PythonExe -m src.research.momentum_factor
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $PythonExe -m src.research.ml_cross_section
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Output "PASS: tests and both research studies completed"
