"""Install Qwen Q4_K_M with SHA256 validation and official Windows Vulkan runtime."""
import hashlib,json,urllib.request,zipfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
MODELS=ROOT/'models'
FILE='Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf'
SOURCE_FILE='Qwen3-4B-Instruct-2507-Q4_K_M.gguf'
REPO='unsloth/Qwen3-4B-Instruct-2507-GGUF'
REVISION='0b0406b39725d752255ffeb48f102c66f98e14aa'
EXPECTED='3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597'
EXPECTED_SIZE=2497281120
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def download(url,path):
    partial=path.with_suffix(path.suffix+'.partial')
    for attempt in range(6):
        try:
            offset=partial.stat().st_size if partial.exists() else 0
            headers={'Range':f'bytes={offset}-'} if offset else {}
            if 'api.github.com' in url:headers['Accept']='application/octet-stream'
            req=urllib.request.Request(url,headers=headers)
            with opener.open(req,timeout=40) as response:
                mode='ab' if response.status==206 else 'wb'
                with partial.open(mode) as f:
                    last=f.tell()
                    while True:
                        chunk=response.read(1024*1024)
                        if not chunk:break
                        f.write(chunk)
                        if f.tell()-last>=64*1024*1024:
                            print(path.name,round(f.tell()/1024**2),'MiB',flush=True);last=f.tell()
            partial.replace(path)
            return
        except Exception as exc:
            print('Download retry',attempt,type(exc).__name__,str(exc),flush=True)
    raise RuntimeError('Download failed: '+url)
def main():
    MODELS.mkdir(exist_ok=True)
    path=MODELS/FILE
    if not path.exists():
        download(f'https://modelscope.cn/models/{REPO}/resolve/{REVISION}/{SOURCE_FILE}',path)
    if path.stat().st_size!=EXPECTED_SIZE:raise ValueError('Incorrect GGUF length')
    digest=hashlib.sha256()
    with path.open('rb') as f:
        if f.read(4)!=b'GGUF':raise ValueError('Invalid GGUF header')
        f.seek(0)
        for chunk in iter(lambda:f.read(8*1024*1024),b''):digest.update(chunk)
    if digest.hexdigest()!=EXPECTED:raise ValueError('GGUF SHA256 mismatch')
    (MODELS/'manifest.json').write_text(json.dumps(dict(source='ModelScope',repo=REPO,revision=REVISION,file=FILE,sha256=EXPECTED,size=EXPECTED_SIZE),indent=2),encoding='utf-8')
    runtime=ROOT/'tools/llama';runtime.mkdir(parents=True,exist_ok=True)
    if not list(runtime.rglob('llama-server.exe')):
        asset={'name':'llama-b12125-bin-win-vulkan-x64.zip','browser_download_url':'https://api.github.com/repos/ggml-org/llama.cpp/releases/assets/712498128?download=1','size':37683656}
        archive=runtime/'runtime.zip'
        if not archive.exists():download(asset['browser_download_url'],archive)
        if hashlib.sha256(archive.read_bytes()).hexdigest()!='29db801266e784d17fbc9dd7ff9b6a350d2b665bdc5658216f37d91c09918f07':raise ValueError('Runtime SHA256 mismatch')
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                if not (runtime/name).resolve().is_relative_to(runtime.resolve()):raise ValueError('Unsafe archive')
            z.extractall(runtime)
        (runtime/'source.json').write_text(json.dumps(asset,indent=2),encoding='utf-8')
    print('Model verified and Vulkan runtime ready.',flush=True)
if __name__=='__main__':main()
