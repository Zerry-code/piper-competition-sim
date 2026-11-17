param([switch]$NoPreview)
$ErrorActionPreference='Stop'
$env:PYTHONUTF8='1'
$workspace=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$env:PYTHONPATH="$workspace\work\runtime;$PSScriptRoot\src;$PSScriptRoot\scripts"
$output=Join-Path $PSScriptRoot 'artifacts\camera-check'
New-Item -ItemType Directory -Path $output -Force | Out-Null
Write-Host '正在检查大白相机的 USB、驱动及图像……'
$present=@(Get-PnpDevice -PresentOnly | Where-Object {$_.InstanceId -match 'VID_2BC5|VID_1D27'})
$all=@(Get-CimInstance Win32_PnPEntity)
$rows=@(foreach($device in $present) {
    $details=$all | Where-Object {$_.DeviceID -eq $device.InstanceId} | Select-Object -First 1
    [pscustomobject]@{Name=$device.FriendlyName;Class=$device.Class;Status=$device.Status;DeviceID=$device.InstanceId;ErrorCode=if($details){[int]$details.ConfigManagerErrorCode}else{-1}}
})
$json=ConvertTo-Json -InputObject $rows -Depth 5
$path=Join-Path $output 'devices.json'
[System.IO.File]::WriteAllText($path,$json,(New-Object System.Text.UTF8Encoding($true)))
$arguments=@((Join-Path $PSScriptRoot 'scripts\check_dabai.py'),'--devices',$path)
if(-not $NoPreview){$arguments+='--preview'}
& python @arguments
exit $LASTEXITCODE
