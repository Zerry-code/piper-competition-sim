"""Sensor-confirmed temporary landing options near the arm's central workspace."""
import numpy as np


def circle_hits_card(center,radius,tray,gap=.015):
    distance=np.maximum(abs(center-np.asarray(tray['center_xy_m']))-np.asarray(tray['half_size_xy_m']),0.)
    return np.linalg.norm(distance)<radius+gap


def floor_coverage(shape,point,seen_plane):
    cells=set()
    for offset,radius in zip(shape.offsets,shape.radii):
        for x in np.arange(-radius,radius+.005,.01):
            for y in np.arange(-radius,radius+.005,.01):
                if x*x+y*y<=radius*radius:
                    cells.add(tuple(np.floor((np.asarray(point)+offset+[x,y])/.01).astype(int)))
    return sum(c in seen_plane for c in cells)/max(1,len(cells))



def staging_surface_safe(shape,point,trays,memory):
    if any(circle_hits_card(np.asarray(point)+offset,r,tray) for offset,r in zip(shape.offsets,shape.radii) for tray in trays.values()):return False
    return not shape.hits_points(point,memory.obstacles) and floor_coverage(shape,point,memory.seen_plane)>=.92

def staging_options(shape,table,trays,memory,reservations,shapes,key,source):
    region=dict(center_xy_m=table['center_xy_m'],half_size_xy_m=table['half_size_xy_m'])
    choices=[]
    for angle in [0.,np.pi/2,-np.pi/2,np.pi/4,-np.pi/4]:
        rotated=shape.rotated(angle)
        for x in [.39,.35,.43,.47]:
            for y in [0.,-.06,.06,-.12,.12]:
                point=np.array([x,y])
                if np.linalg.norm(point-np.asarray(source)[:2])<.06:continue
                if rotated.clearance(region,point)<.025:continue
                if any(circle_hits_card(point+offset,r,tray) for offset,r in zip(rotated.offsets,rotated.radii) for tray in trays.values()):continue
                if any(k!=key and k in shapes and rotated.overlaps(point,shapes[k],p) for k,p in reservations.items()):continue
                if rotated.hits_points(point,memory.obstacles):continue
                coverage=floor_coverage(rotated,point,memory.seen_plane)
                if coverage<.92:continue
                score=np.linalg.norm(point-[.39,0])+.08*abs(angle)+.1*np.linalg.norm(point-np.asarray(source)[:2])
                choices.append(dict(point=point,rotation=angle,shape=rotated,coverage=coverage,score=float(score)))
    return sorted(choices,key=lambda o:o['score'])


def select_staging_landings(options,source,limit=6):
    """Keep central and short-transfer alternatives, rather than six neighbours."""
    if not options:return []
    angle=options[0]['rotation'];pool=[o for o in options if abs(o['rotation']-angle)<1e-6]
    seeds=[pool[0],min(pool,key=lambda o:np.linalg.norm(o['point']-np.asarray(source)[:2])),
           max(pool,key=lambda o:o['point'][0]),min(pool,key=lambda o:o['point'][1]),max(pool,key=lambda o:o['point'][1])]
    chosen=[]
    for o in seeds+pool:
        if not any(np.linalg.norm(o['point']-old['point'])<.005 for old in chosen):chosen.append(o)
        if len(chosen)>=limit:break
    return chosen


def progressive_options(shape,table,trays,memory,reservations,shapes,key,source,goal,destination):
    """Greedy, first-leg-only landings: prefer the largest safe goalward advance."""
    source=np.asarray(source)[:2];goal=np.asarray(goal)[:2];delta=goal-source
    distance=np.linalg.norm(delta)
    if distance<.015:return []
    normal=np.array([-delta[1],delta[0]])/distance
    region=dict(center_xy_m=table['center_xy_m'],half_size_xy_m=table['half_size_xy_m'])
    other_cards={c:t for c,t in trays.items() if c!=destination}
    candidates=[]
    points=[goal.copy()]
    for fraction in [1.,.85,.65,.45,.25]:
        for side in [0.,.035,-.035,.07,-.07]:points.append(source+fraction*delta+side*normal)
    for x in np.arange(.24,.561,.04):
        for y in np.arange(-.34,.341,.04):points.append(np.array([x,y]))
    for angle in [0.,np.pi/2,-np.pi/2,np.pi/4,-np.pi/4,np.pi]:
        rotated=shape.rotated(angle)
        for point in points:
            remaining=np.linalg.norm(point-goal);advance=distance-remaining
            if advance<min(.02,distance*.2) or np.linalg.norm(point-source)<.025:continue
            if not (.20<=point[0]<=.90 and np.linalg.norm(point)<=.95):continue
            if rotated.clearance(region,point)<.025:continue
            if any(k!=key and k in shapes and rotated.overlaps(point,shapes[k],p) for k,p in reservations.items()):continue
            if not staging_surface_safe(rotated,point,other_cards,memory):continue
            # Progress dominates distance/turn cost; direct goal is tried first.
            score=remaining+.05*abs(angle)+.05*np.linalg.norm(point-source)
            candidates.append(dict(point=np.asarray(point),rotation=angle,shape=rotated,coverage=floor_coverage(rotated,point,memory.seen_plane),score=float(score),goal_distance=float(remaining),advance=float(advance)))
    return sorted(candidates,key=lambda o:o['score'])


def select_progressive_landings(options,source,goal,limit=6):
    if not options:return []
    angle=options[0]['rotation'];pool=[o for o in options if abs(o['rotation']-angle)<1e-6]
    source=np.asarray(source)[:2];goal=np.asarray(goal)[:2]
    seeds=[pool[0]]
    for fraction in [.8,.6,.4,.2]:
        desired=source+fraction*(goal-source)
        seeds.append(min(pool,key=lambda o:np.linalg.norm(o['point']-desired)))
    selected=[]
    for o in seeds+pool:
        if all(np.linalg.norm(o['point']-p['point'])>.025 for p in selected):selected.append(o)
        if len(selected)>=limit:break
    return sorted(selected,key=lambda o:o['goal_distance'])
