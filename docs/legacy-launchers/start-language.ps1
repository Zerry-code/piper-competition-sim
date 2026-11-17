$ErrorActionPreference='Stop'
try {
    $request=[System.Net.WebRequest]::Create('http://127.0.0.1:1234/health')
    $request.Proxy=$null
    $request.Timeout=1000
    $response=$request.GetResponse()
    $response.Close()
    Write-Host 'Local model service is already running on port 1234.'
    exit 0
} catch {}

$server=Get-ChildItem -LiteralPath "$PSScriptRoot\tools\llama" -Filter llama-server.exe -Recurse | Select-Object -First 1
if (-not $server) { throw 'Run python scripts/setup_language.py first.' }
$modelPath=python -c "import ctypes,sys;b=ctypes.create_unicode_buffer(32768);ctypes.windll.kernel32.GetShortPathNameW(sys.argv[1],b,len(b));print(b.value)" "$PSScriptRoot\models\Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
& $server.FullName -m $modelPath --alias Qwen3-4B-Instruct-2507 --host 127.0.0.1 --port 1234 -c 4096 -ngl 99 --parallel 1
