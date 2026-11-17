param([Nullable[int]]$Seed=$null)
$ErrorActionPreference='Stop'
$workspace=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$env:PYTHONUTF8='1'
$env:PYTHONPATH="$workspace\work\runtime;$PSScriptRoot\src"
$colors=@{'1'='red';'2'='green';'3'='yellow'}
Write-Host 'L1: 50mm cubes -> 300mm flat cards. 1=Red 2=Green 3=Yellow'
$source=Read-Host 'Cube color (1/2/3)'
$destination=Read-Host 'Target card color (1/2/3)'
if (-not $colors.ContainsKey($source) -or -not $colors.ContainsKey($destination)) { throw 'Choose 1, 2 or 3 for each color.' }
$arguments=@("$PSScriptRoot\scripts\run_l1.py",'--viewer','--source',$colors[$source],'--destination',$colors[$destination])
if ($null -ne $Seed) { $arguments+=@('--seed',"$Seed") }
python @arguments
exit $LASTEXITCODE
