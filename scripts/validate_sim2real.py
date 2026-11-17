"""Same-seed closed-loop trials; keep sensor completion and truth evaluation separate."""
import json
from web_console import Console,ROOT

def main():
    output=ROOT/'artifacts/sim2real-validation';output.mkdir(exist_ok=True)
    results=[]
    for profile in ['baseline','mild','stress']:
        app=Console(output_root=output/profile);app.gap_profile=profile
        try:
            app.reset('l1',2351137725)
            app.callback=lambda skills,original,stage:original(stage)
            app.execute('把红色方块放到黄色卡纸上')
            result=dict(profile=profile,result=app.data['result'],evaluation=app.world.evaluation(),gap=app.world.deployment_report)
        except Exception as exc:result=dict(profile=profile,error=f'{type(exc).__name__}: {exc}',evaluation=app.world.evaluation() if app.world else None)
        finally:
            if app.renderer:app.renderer.close()
            if app.world:app.world.close()
        results.append(result)
        (output/'result.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(result,ensure_ascii=True),flush=True)
if __name__=='__main__':main()
