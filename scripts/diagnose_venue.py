from web_console import Console,ROOT
from piper_sim.l1_world import L1World
import numpy as np
orig=L1World._valid_reset
L1World._valid_reset=lambda s:True
app=Console(output_root=ROOT/'artifacts/venue-debug');app.reset('l1',721)
w=app.world;b=w.backend
print('valid',orig(w),'table',w.cfg['table'])
for o in w.objects:print(o['id'],w.bounds(o['id']))
for c in b.data.contact[:b.data.ncon]:
 print(b.model.geom(c.geom1).name,b.model.geom(c.geom2).name,c.dist)
app.renderer.close();w.close()
