$ErrorActionPreference='Stop'
function Test-LocalModel {
    try { $request=[System.Net.WebRequest]::Create('http://127.0.0.1:1234/health'); $request.Proxy=$null; $request.Timeout=1000; $response=$request.GetResponse(); $response.Close(); return $true } catch { return $false }
}
if (-not (Test-LocalModel)) {
    $launcher=Join-Path $PSScriptRoot 'start-language.ps1'
    $logDirectory=Join-Path $PSScriptRoot 'artifacts'
    Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',('"'+$launcher+'"')) -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logDirectory 'model-service.log') -RedirectStandardError (Join-Path $logDirectory 'model-service-errors.log')
    $ready=$false
    for($attempt=0;$attempt -lt 60;$attempt++) { if(Test-LocalModel){$ready=$true;break};Start-Sleep -Milliseconds 500 }
    if(-not $ready){throw '本地模型未能启动，请查看 artifacts 中的 model-service-errors.log。'}
}
$instruction=Read-Host '输入中文任务（例如：把所有方块按颜色分类，或把香蕉放入绿色托盘）'
if ([string]::IsNullOrWhiteSpace($instruction)) { exit }
& "$PSScriptRoot\run-harvest.ps1" -Instruction $instruction -Viewer
