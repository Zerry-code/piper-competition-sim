import json,numpy as np
from web_console import Console,ROOT,WebFruitSkills

def main():
    app=Console(output_root=ROOT/'artifacts/fruit-space-diagnosis');app.gap_profile='stress';app.reset('l2',732000359)
    skills=WebFruitSkills(app.world,app.output,dict(category='all',destination='mapping',assignments=dict(banana='yellow',persimmon='yellow',cucumber='green',pear='green',bell_pepper='red',potato='red')));skills.record_stage_images=False
    try:
        try:
            tracks=skills.overview()
            report=dict(status='allocated',count=len(tracks),items={t.key:dict(destination=t.destination,position=t.destination_position.tolist(),extent=np.ptp(t.estimate.points,axis=0).tolist()) for t in tracks})
        except Exception as exc:report=dict(status='failed',error=str(exc))
        (ROOT/'artifacts/fruit-space-diagnosis/result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(report,ensure_ascii=True),flush=True)
    finally:app.renderer.close();app.world.close()
if __name__=='__main__':main()
