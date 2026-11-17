"""Soft end-effector approach preference; no elbow-height or joint-count objective."""
import numpy as np

def natural_tilts(values):
    values=list(dict.fromkeys(values))
    def cost(tilt):
        # Prefer moderate forward/downward approach, but keep every supplied option.
        upright_penalty=2.*max(0.,-float(np.cos(tilt)))
        reverse_penalty=.6 if tilt<0 else 0.
        return abs(float(tilt)-.35)+reverse_penalty+upright_penalty
    return sorted(values,key=cost)


def fair_tilts(values):
    """Prefer forward poses, while regularly reserving trials for other poses."""
    ordered=natural_tilts(values)
    forward=[t for t in ordered if 0<=t<=1.25]
    fallback=[t for t in ordered if t not in forward]
    result=[]
    while forward or fallback:
        result+=forward[:2];forward=forward[2:]
        if fallback:result.append(fallback.pop(0))
    return result


def placement_orientations(tilts,yaws,grasp_tilt,grasp_yaw):
    """Keep carried tilt stable; upright placement is a soft secondary cost."""
    def score(pose):
        tilt,yaw=pose
        natural=abs(tilt-.35)+(.6 if tilt<0 else 0.)+2*max(0.,-np.cos(tilt))
        turn=abs(np.arctan2(np.sin(yaw-grasp_yaw),np.cos(yaw-grasp_yaw)))
        return 1.5*abs(tilt-grasp_tilt)+.25*turn+.30*natural
    return sorted([(t,y) for t in dict.fromkeys(tilts) for y in dict.fromkeys(yaws)],key=score)
