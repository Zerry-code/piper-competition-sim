param([ValidateSet('l1','l2')][string]$Mode='l1')
$ErrorActionPreference='Stop'
$workspace=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$env:PYTHONUTF8='1'
$env:PYTHONPATH="$workspace\work\runtime;$PSScriptRoot\src"
function Test-LocalService([string]$Url) {
    try {
        $probe=[System.Net.WebRequest]::Create($Url);$probe.Proxy=$null;$probe.Timeout=1500
        $response=$probe.GetResponse();$response.Close();return $true
    } catch { return $false }
}
$logs=Join-Path $PSScriptRoot 'artifacts\startup'
New-Item -ItemType Directory -Path $logs -Force | Out-Null
Write-Host '正在检查网页操作台，请稍候……'
if (-not (Test-LocalService 'http://127.0.0.1:1234/health')) {
    Write-Host '中文模型正在后台加载；操作台可先打开。'
    Start-Process powershell -WindowStyle Hidden -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',('"'+$PSScriptRoot+'\start-language.ps1"')) -RedirectStandardOutput (Join-Path $logs 'language-output.log') -RedirectStandardError (Join-Path $logs 'language-error.log')
}
if (-not (Test-LocalService 'http://127.0.0.1:8772/api/state')) {
    $python=(Get-Command python -ErrorAction Stop).Source
    $process=Start-Process -FilePath $python -WindowStyle Hidden -WorkingDirectory $PSScriptRoot -ArgumentList @(('"'+$PSScriptRoot+'\scripts\web_console.py"'),'--mode',$Mode,'--no-browser') -RedirectStandardOutput (Join-Path $logs 'web-output.log') -RedirectStandardError (Join-Path $logs 'web-error.log') -PassThru
    $ready=$false
    for ($attempt=0;$attempt -lt 90;$attempt++) {
        if (Test-LocalService 'http://127.0.0.1:8772/api/state') {$ready=$true;break}
        if ($process.HasExited) {throw ('操作台启动失败，请查看：'+(Join-Path $logs 'web-error.log'))}
        Start-Sleep -Milliseconds 500
    }
    if (-not $ready) {throw ('操作台启动超时，请查看：'+(Join-Path $logs 'web-error.log'))}
}
Write-Host '操作台已启动：http://127.0.0.1:8772'
Start-Process 'http://127.0.0.1:8772'
