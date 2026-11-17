"""Procedural implicit ellipsoid-union fruit. Not real scans or calibrated fruit physics.

The same implicit components define the visual isosurface and compound collision body.
Each ellipsoid is convex; their union preserves a banana's concavity.
"""
from pathlib import Path
import numpy as np
from skimage.measure import marching_cubes
from PIL import Image
from scipy.ndimage import gaussian_filter

FRUITS = ('apple', 'pear', 'banana', 'lemon', 'orange', 'strawberry', 'peach', 'plum','persimmon','bell_pepper','cucumber','potato')
COLORS = {'apple': [.85,.38,.34,1], 'pear': [.4,.65,.12,1],
          'banana': [.95,.8,.08,1], 'lemon': [.8,.85,.15,1],
          'orange': [.95,.35,.05,1], 'strawberry': [.65,.03,.18,1],
          'peach': [.95,.45,.45,1], 'plum': [.3,.08,.45,1], 'persimmon':[.95,.32,.035,1], 'bell_pepper':[.75,.04,.025,1], 'cucumber':[.055,.27,.035,1], 'potato':[.48,.34,.19,1]}


def components(category, rng):
    scale = rng.uniform(.85, 1.12, 3)
    if category == 'banana':
        ts = np.linspace(-1., 1., 33)
        straight_fraction=rng.uniform(.30,.50)
        curvature=rng.uniform(.020,.035)
        # Variable-length cylindrical middle with curved ends; some samples remain
        # too curved for the wide flat pads and must be rejected by grasp selection.
        parts = [(np.array([.055*t, curvature*(max(abs(t)-straight_fraction,0)/(1-straight_fraction))**2-curvature/2, 0]),
                  np.array([.011,.011,.012])*(.45+.55*(1-t*t))) for t in ts]
    elif category == 'bell_pepper':
        parts=[(np.array([.012*np.cos(a),.012*np.sin(a),0]),np.array([.018,.018,.027])) for a in np.linspace(0,2*np.pi,4,endpoint=False)]
    elif category == 'cucumber':
        parts=[(np.array([t,.004*np.sin(t*30),0]),np.array([.015,.013,.013])*(.8+.2*np.cos(t*22))) for t in np.linspace(-.043,.043,13)]
    elif category == 'potato':
        parts=[(np.array([0,0,0]),np.array([.031,.023,.021])),(np.array([.019,.002,0]),np.array([.019,.019,.018]))]
    elif category == 'persimmon':
        parts=[(np.zeros(3),np.array([.029,.028,.023]))]
    elif category == 'pear':
        parts = [(np.array([0,0,-.006]), np.array([.024,.024,.026])),
                 (np.array([0,0,.019]), np.array([.014,.014,.022]))]
    elif category == 'strawberry':
        parts=[(np.array([0,0,z]),np.array([r,r,h])) for z,r,h in
               [(-.016,.009,.010),(-.007,.015,.013),(.004,.020,.015),(.013,.018,.012)]]
    else:
        radii = {'apple':[.027,.026,.024], 'lemon':[.032,.022,.022],
                 'orange':[.027,.027,.026], 'strawberry':[.020,.020,.026],
                 'peach':[.027,.026,.025], 'plum':[.022,.021,.025]}[category]
        parts = [(np.zeros(3), np.array(radii))]
    return [(p*scale, r*scale) for p,r in parts]


def field(points, parts):
    return np.min(np.stack([np.linalg.norm((points-p)/r, axis=-1)-1
                            for p,r in parts]), axis=0)


def mesh(parts, path, resolution=64):
    low = np.min([p-r for p,r in parts],0)-.002
    high = np.max([p+r for p,r in parts],0)+.002
    axes = [np.linspace(a,b,resolution) for a,b in zip(low, high)]
    xyz = np.stack(np.meshgrid(*axes,indexing='ij'),-1)
    vertices, faces, normals, _ = marching_cubes(field(xyz,parts), 0,
                                          spacing=(high-low)/(resolution-1))
    vertices += low
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open('w') as f:
        for v in vertices: f.write('v '+' '.join(map(str,v))+'\n')
        center=(low+high)/2
        normalized=(vertices-center)/(high-low)
        for v in normalized:
            u=np.arctan2(v[1],v[0])/(2*np.pi)+.5
            w=np.arctan2(v[2],np.linalg.norm(v[:2]))/np.pi+.5
            f.write(f'vt {u} {w}\n')
        for n in normals:f.write('vn '+' '.join(map(str,-n))+'\n')
        for face in faces+1: f.write('f '+' '.join(f'{i}/{i}/{i}' for i in face)+'\n')
    return low+.002, high-.002


def texture(category,path,rng):
    """Procedural skin variation; no real fruit data or simulator labels in observations."""
    size=512
    y,x=np.indices((size,size))
    color=np.array(COLORS[category][:3])*255
    noise=gaussian_filter(rng.normal(0,1,(size,size)),2)
    noise/=max(noise.std(),1e-6)
    pixels=np.broadcast_to(color,(size,size,3)).copy()
    pixels*=np.clip(1+.045*noise[...,None],.7,1.2)
    if category=='cucumber':
        pixels*=1+.10*np.sin(x/size*16*np.pi)[...,None]
        pixels[rng.random((size,size))<.018]*=.45
    elif category=='potato':
        pixels[rng.random((size,size))<.025]*=[.45,.38,.30]
        pixels*=np.clip(1+.06*noise[...,None],.6,1.3)
    if category in ('apple','peach'):
        blush=.12*(1+np.sin(x/size*4*np.pi+.4*np.sin(y/50)))
        pixels[...,0]*=1+blush
        pixels[...,1:]*=1-blush[...,None]*.5
    if category=='banana':
        streak=(np.sin(x/9)+np.sin(x/19))*.018
        pixels*=1+streak[...,None]
        spots=gaussian_filter(rng.random((size,size)),1)<.24
        pixels[spots]*=[.40,.27,.16]
    elif category=='orange':
        pixels*=1+.07*rng.normal(0,1,(size,size,1))
    elif category=='strawberry':
        for row in range(25,size,37):
            for col in range(15+(row//37%2)*16,size,35):
                seed=((x-col)/2.8)**2+((y-row)/4.5)**2<1
                pixels[seed]=[220,178,72]
    else:
        freckles=rng.random((size,size))<.008
        pixels[freckles]*=.60
    Image.fromarray(np.clip(pixels,0,255).astype(np.uint8)).save(path)
