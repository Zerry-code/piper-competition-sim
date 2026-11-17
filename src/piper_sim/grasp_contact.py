"""Simulation tactile proxy: opposing normals, pad-centred contact and finite force."""
import numpy as np
import mujoco


def contact_quality(backend):
    model,data=backend.model,backend.data
    samples=[]
    warnings=[]
    for name in ['link7_pad','link8_pad']:
        gid=model.geom(name).id
        entries=[]
        for i,c in enumerate(data.contact[:data.ncon]):
            if gid not in (c.geom1,c.geom2):continue
            other=c.geom2 if c.geom1==gid else c.geom1
            bid=int(model.geom_bodyid[other])
            if model.body_dofnum[bid]!=6:continue
            force=np.zeros(6);mujoco.mj_contactForce(model,data,i,force)
            if force[0]<.15:continue
            normal=np.array(c.frame[:3])*(1 if c.geom1==gid else -1)
            local=data.geom_xmat[gid].reshape(3,3).T@(c.pos-data.geom_xpos[gid])
            size=model.geom_size[gid]
            central=bool(np.all(abs(local[:2])<.85*size[:2]))
            face=abs(np.dot(normal,data.geom_xmat[gid].reshape(3,3)[:,2]))>.90
            # Edge location is a planning/quality warning, not proof that a physically
            # closed gripper holds nothing. Keep real bilateral contacts for verification.
            if not central:warnings.append('contact_near_pad_edge')
            if not face:warnings.append('oblique_contact')
            entries.append((bid,normal,float(force[0])))
        samples.append(entries)
    common={x[0] for x in samples[0]}&{x[0] for x in samples[1]}
    for bid in common:
        normals=[]
        for side in samples:
            n=sum((normal*force for body,normal,force in side if body==bid),np.zeros(3))
            normals.append(n/max(np.linalg.norm(n),1e-12))
        opposition=float(np.dot(*normals))
        if opposition<-.70:return dict(accepted=True,loaded=True,normal_dot=opposition,min_force_n=min(sum(force for body,normal,force in side if body==bid) for side in samples),warnings=sorted(set(warnings)))
    return dict(accepted=False,loaded=bool(common),reason='Missing loaded bilateral opposing contact',warnings=sorted(set(warnings)))


def retained_support(backend):
    """Tactile proxy: the body touching both pads is supported by table/card."""
    model,data=backend.model,backend.data
    pads={model.geom(n).id for n in ('link7_pad','link8_pad')}
    sides={g:set() for g in pads}
    for contact in data.contact[:data.ncon]:
        for pad in pads:
            if pad in (contact.geom1,contact.geom2):
                other=contact.geom2 if contact.geom1==pad else contact.geom1
                bid=int(model.geom_bodyid[other])
                if model.body_dofnum[bid]==6:sides[pad].add(bid)
    held=set.intersection(*sides.values())
    if not held:return False
    for index,contact in enumerate(data.contact[:data.ncon]):
        for item,support in ((contact.geom1,contact.geom2),(contact.geom2,contact.geom1)):
            name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(support)) or ''
            if int(model.geom_bodyid[item]) in held and (name in ('competition_table','installation_table') or name.startswith('tray_') and name.endswith('_floor')):
                force=np.zeros(6);mujoco.mj_contactForce(model,data,index,force)
                if force[0]>.05:return True
    return False
