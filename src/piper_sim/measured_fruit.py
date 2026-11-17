"""Measured approximate body sizes, metres; ellipsoid sections use Ramanujan perimeter."""
import numpy as np
from scipy.optimize import brentq

def ellipsoid_axes(long_circ,short_circ):
    b=short_circ/(2*np.pi)
    def perimeter(a):
        h=((a-b)/(a+b))**2
        return np.pi*(a+b)*(1+3*h/(10+np.sqrt(4-3*h)))
    a=brentq(lambda a:perimeter(a)-long_circ,b,b*5)
    return np.array([2*a,2*b,2*b])

def measured_parts(category,parts):
    import yaml
    from piper_sim.calibration import ROOT
    props=yaml.safe_load((ROOT/'config/fruit_props.yaml').read_text(encoding='utf-8-sig'))
    widths=props['unmeasured_width_m']
    dimensions={'pear':ellipsoid_axes(.33,.27),'potato':ellipsoid_axes(.26,.215),
        'apple':np.array([.26/np.pi,.26/np.pi,.07]),'bell_pepper':np.array([.26/np.pi,.26/np.pi,.08]),
        'persimmon':np.array([.24/np.pi,.24/np.pi,.05])}
    if category in ('pear','potato'):
        size=dimensions[category]/2
        parts=[(np.zeros(3),size)]
        if category=='pear':
            parts.extend((np.array([size[0]+.0025+i*.005,0,0]),np.array([.0025,.002,.002])) for i in range(5))
        return parts
    low=np.min([p-r for p,r in parts],axis=0);high=np.max([p+r for p,r in parts],axis=0)
    target=dimensions.get(category)
    if target is None:
        target=high-low
        if category=='cucumber':target[0]=.26;target[1:]=[widths['cucumber']]*2
        elif category=='banana':
            # End-to-end chord is the centreline endpoint distance; retain bend/thickness.
            chord=np.linalg.norm(parts[-1][0]-parts[0][0]);target[0]*=.18/chord
            target[1]=max(target[1],widths['banana']);target[2]=widths['banana']*.875
        else:return parts
    scale=target/(high-low)
    return [(p*scale,r*scale) for p,r in parts]
