param([string]$Python = '')
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$portal = Split-Path -Parent $PSScriptRoot
if (-not $Python) { $Python = Join-Path $portal '.venv\Scripts\python.exe' }
& $Python (Join-Path $portal 'serve.py')
exit $LASTEXITCODE
