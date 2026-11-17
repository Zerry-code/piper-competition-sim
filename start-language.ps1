$ErrorActionPreference='Stop'
$server=Get-ChildItem -LiteralPath (Join-Path $PSScriptRoot 'tools\llama') -Filter llama-server.exe -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
$model=Join-Path $PSScriptRoot 'models\Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf'
if (-not $server -or -not (Test-Path -LiteralPath $model)) { throw 'Model/runtime not bundled. Run scripts/setup_language.py explicitly, or supply a compatible local service; see README.md.' }
& $server.FullName -m $model --alias Qwen3-4B-Instruct-2507 --host 127.0.0.1 --port 1234 -c 4096 -ngl 99 --parallel 1
