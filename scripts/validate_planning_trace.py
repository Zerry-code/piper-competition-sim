"""Reported cucumber layout: inspect planning evidence without attempting a grasp."""
import json,threading,urllib.request
import numpy as np
from web_console import Console,ROOT,WebFruitSkills,server

def main():
    output=ROOT/'artifacts/planning-search-validation';app=Console(output_root=output);app.reset('l2',1090030603)
    task=json.loads((ROOT/'artifacts/web-console/scene-1-1090030603/task-2.json').read_text(encoding='utf-8'))['parsed']
    skills=WebFruitSkills(app.world,app.output,task);app.bind_callback(skills)
    try:
        targets=skills.overview();target=next(t for t in targets if t.category=='cucumber')
        before=app.world.backend.measured().copy();time_before=app.world.backend.data.time
        try:skills.prepare_target(target,target.estimate)
        except RuntimeError:pass
        report=app.last_planning_report
        assert report and report['events'] and report['target']=='黄瓜'
        np.testing.assert_equal(before,app.world.backend.measured());assert time_before==app.world.backend.data.time
        http=server(app,0);worker=threading.Thread(target=http.serve_forever,daemon=True);worker.start()
        try:
            op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
            data=json.load(op.open('http://127.0.0.1:'+str(http.server_port)+'/api/planning-report'))
            assert data['status']==report['status'] and data['attempts']==report['attempts']
        finally:http.shutdown();http.server_close();worker.join()
        (output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(dict(status=report['status'],attempts=report['attempts'],unique=report['unique_combinations'],scope=report['scope'],phase=report['current'].get('phase_label'),no_robot_motion=True,download_verified=True),ensure_ascii=True),flush=True)
    finally:app.renderer.close();app.world.close()
if __name__=='__main__':main()
