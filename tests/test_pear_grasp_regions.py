import numpy as np
from piper_sim.grasp_geometry import pear_grasp_regions

def test_sections_avoid_thin_neck_and_offer_multiple_directions():
    points=[]
    for z in np.linspace(.075,.135,45):
        radius=.025 if z<.112 else .009
        for angle in np.linspace(0,2*np.pi,100,endpoint=False):
            points.append([.4+radius*np.cos(angle),.1+radius*np.sin(angle),z])
    options=pear_grasp_regions(np.array(points),.06)
    assert options
    assert all(o['center'][2]<.12 for o in options)
    assert len({round(o['yaw'],2) for o in options})>4
    assert len({round(o['center'][2],3) for o in options})>1
    assert all(o['opening_m']<.066 for o in options)
