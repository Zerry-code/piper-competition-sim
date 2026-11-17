import numpy as np
from piper_sim.fruit_perception import classify_visible_geometry


def test_yellow_textured_banana_is_not_a_yellow_cube():
    x=np.linspace(-.06,.06,80);points=np.array([[u,.55*u*u/.06+w,.02] for u in x for w in np.linspace(-.008,.008,8)])
    name,features=classify_visible_geometry(points,np.tile([107,88,6],(len(points),1)),0.)
    assert name=='banana' and not features['cube_like']


def test_yellow_green_pear_is_not_cube_or_banana():
    a,b=np.meshgrid(np.linspace(0,np.pi/2,25),np.linspace(0,2*np.pi,50))
    points=np.c_[.025*np.sin(a.ravel())*np.cos(b.ravel()),.037*np.sin(a.ravel())*np.sin(b.ravel()),.025+.024*np.cos(a.ravel())]
    name,features=classify_visible_geometry(points,np.tile([79,92,9],(len(points),1)),0.)
    assert name=='pear' and not features['cube_like']


def test_real_flat_yellow_cube_remains_recognized():
    x,y=np.meshgrid(np.linspace(-.025,.025,25),np.linspace(-.025,.025,25));points=np.c_[x.ravel(),y.ravel(),np.full(x.size,.05)]
    name,features=classify_visible_geometry(points,np.tile([180,150,6],(len(points),1)),0.)
    assert name=='cube_yellow' and features['cube_like']


def test_oblique_cube_remains_cube_when_side_pixels_dominate():
    x,y=np.meshgrid(np.linspace(-.025,.025,25),np.linspace(-.025,.025,25))
    top=np.c_[x.ravel(),y.ravel(),np.full(x.size,.05)]
    z,v=np.meshgrid(np.linspace(.007,.05,45),np.linspace(-.025,.025,35))
    sides=np.c_[np.full(z.size,.025),v.ravel(),z.ravel()]
    points=np.vstack([top,sides])
    name,features=classify_visible_geometry(points,np.tile([30,100,8],(len(points),1)),0.)
    assert features['flat_fraction']<.55
    assert features['cube_like'] and name=='cube_green'
