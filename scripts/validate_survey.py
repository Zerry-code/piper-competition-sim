import json
from web_console import Console,ROOT,WebFruitSkills

def main():
    results=[]
    for seed in [1632549805,718823998,732000359]:
        app=Console(output_root=ROOT/'artifacts/three-view-survey');app.reset('l2',seed)
        skills=WebFruitSkills(app.world,app.output,dict(category='all',destination='mapping'));skills.record_stage_images=False
        try:
            try:
                tracks=skills.overview();r=dict(seed=seed,count=len(tracks),survey=[e for e in skills.navigation_events if e.get('event')=='three_view_survey'])
            except Exception as exc:r=dict(seed=seed,error=str(exc))
            results.append(r);print(json.dumps(r),flush=True)
        finally:app.renderer.close();app.world.close()
    (ROOT/'artifacts/three-view-survey/result.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
if __name__=='__main__':main()
