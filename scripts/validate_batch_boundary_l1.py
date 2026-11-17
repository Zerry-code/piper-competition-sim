"""Real web-console L1 batch: local language parser, all placements, then one audit."""
import json,time
from web_console import Console,ROOT


def main():
    output=ROOT/'artifacts/batch-boundary-l1-validation';app=Console(output_root=output);app.reset('l1',2329111945)
    # Headless QA keeps the physical controller, omits display refresh/sleep only.
    def callback(skills,original,stage):original(stage);app.stage(stage)
    app.callback=callback
    start=time.monotonic()
    try:
        app.execute('把三个颜色的方块分别放入对应颜色的区域。')
        report=dict(result=app.data.get('result'),record=app.current_record,evaluation=app.world.evaluation(),wall_s=time.monotonic()-start)
        (output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(report['result'],ensure_ascii=True),flush=True)
    finally:app.renderer.close();app.world.close()

if __name__=='__main__':main()
