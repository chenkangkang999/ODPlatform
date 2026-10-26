param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$portal = Split-Path -Parent $PSScriptRoot
& $Python -c "import sys; assert sys.version_info >= (3,12), 'Web环境需要Python3.12或更新版本'"
if ($LASTEXITCODE -ne 0) { throw 'Python版本不匹配' }
$venv = Join-Path $portal '.venv'
if (-not (Test-Path -LiteralPath $venv)) {
    & $Python -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw '创建独立环境失败' }
}
$runtime = Join-Path $venv 'Scripts\python.exe'
& $runtime -m pip install -r (Join-Path $portal 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Web依赖安装失败' }
& $runtime -m pip check
if ($LASTEXITCODE -ne 0) { throw '依赖检查失败' }
Write-Output "Web环境: $runtime"
Write-Output '本脚本不安装或升级同学的算法环境，不复制API密钥，不提交Git。'
