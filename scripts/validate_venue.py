"""Build both task venues and render measured table/card dimensions."""
import json
import mujoco
import numpy as np
from PIL import Image
from web_console import Console,ROOT
out=ROOT/'artifacts/venue-validation';app=Console(output_root=out);report=[]
try:
    for mode in ['l1','l2']:
        app.reset(mode,721)
        m=app.world.backend.model;d=app.world.backend.data
        for name,width in [('installation_table',1.),('competition_table',1.2)]:
            g=m.geom(name);assert abs(g.size[1]*2-width)<1e-8
            assert abs(g.pos[2]+g.size[2])<1e-8
        a=m.geom('installation_table');b=m.geom('competition_table')
        assert abs(a.size[0]*2-.495)<1e-8
        vertices=[]
        for g in range(m.ngeom):
            if m.geom_bodyid[g]!=m.body('base_link').id or m.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH:continue
            mid=m.geom_dataid[g];start=m.mesh_vertadr[mid];count=m.mesh_vertnum[mid]
            points=m.mesh_vert[start:start+count]@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
            vertices.extend(points[:,0])
        base_rear=float(min(vertices));table_rear=float(a.pos[0]-a.size[0])
        assert abs(base_rear-table_rear)<1e-7
        assert abs(b.size[0]*2-.405)<1e-8
        assert abs((b.pos[0]+b.size[0])-(a.pos[0]-a.size[0])-.9)<1e-8
        for color in app.world.cfg['trays']:
            g=m.geom('tray_'+color+'_floor')
            assert list(g.size[:2])==[.1,.1]
            assert abs((d.geom_xpos[g.id,0]-g.size[0])-(a.pos[0]+a.size[0])-.05)<1e-8
        camera=mujoco.MjvCamera();camera.lookat[:]=[.40,0,0];camera.distance=1.75;camera.azimuth=135;camera.elevation=-55
        app.renderer.update_scene(d,camera=camera)
        Image.fromarray(app.renderer.render()).save(out/(mode+'-venue.png'))
        report.append(dict(mode=mode,seed=721,reset_attempt=app.world.reset_attempt,dimensions_verified=True,base_rear_m=base_rear,table_rear_m=table_rear,objects=len(app.world.objects)))
finally:
    if app.renderer:app.renderer.close()
    if app.world:app.world.close()
    (out/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(report)
