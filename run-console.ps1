param([ValidateSet('l1','l2')][string]$Mode='l1')
$ErrorActionPreference='Stop'
$python=if ($env:PIPER_PYTHON) { $env:PIPER_PYTHON } else { Join-Path $PSScriptRoot '.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $python)) { throw 'Create .venv and install requirements-local.txt first; see README.md.' }
$env:PYTHONUTF8='1'
$env:PYTHONPATH="$PSScriptRoot\src;$PSScriptRoot\scripts"
Push-Location $PSScriptRoot
try {
    & $python (Join-Path $PSScriptRoot 'scripts\prepare_assets.py')
    if ($LASTEXITCODE -ne 0) { throw 'Asset preparation failed.' }
    Write-Host 'Console: http://127.0.0.1:8772 ; language service must be available on port 1234 before executing a task.'
    & $python (Join-Path $PSScriptRoot 'scripts\web_console.py') --mode $Mode
    if ($LASTEXITCODE -ne 0) { throw 'Console exited with an error; read the terminal traceback.' }
} finally { Pop-Location }
