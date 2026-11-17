param([Nullable[int]]$Seed=$null,[ValidateSet('fruits','blocks')][string]$Mode='fruits',[string]$Fruits='apple,pear,banana,lemon,orange,strawberry',[string]$Instruction='',[switch]$ObserveOnly,[switch]$Viewer,[Nullable[double]]$TimeLimit=$null)
$ErrorActionPreference='Stop'
$workspace=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$env:PYTHONUTF8='1'
$env:PYTHONPATH="$workspace\work\runtime;$PSScriptRoot\src"
$arguments=@("$PSScriptRoot\scripts\run_harvest.py",'--mode',$Mode,'--fruits',$Fruits)
if ($null -ne $Seed) { $arguments+=@('--seed',"$Seed") }
if ($Instruction) { $arguments+=@('--instruction',$Instruction) }
if ($ObserveOnly) { $arguments+='--observe-only' }
if ($Viewer) { $arguments+='--viewer' }
if ($null -ne $TimeLimit) { $arguments+=@('--time-limit',"$TimeLimit") }
python @arguments
exit $LASTEXITCODE
