"""C2 joint paths with zero endpoint velocity/acceleration and bounded rates."""
import numpy as np
from scipy.interpolate import CubicSpline


def trajectory(waypoints,width_start,width_end,interface,max_acceleration=.45):
    points=np.asarray(waypoints,dtype=float)
    # Knot spacing follows joint distance, not the arbitrary density of path samples.
    lengths=np.linalg.norm(np.diff(points,axis=0),axis=1)
    points=points[np.r_[True,lengths>1e-8]]
    if len(points)<2:points=np.vstack([points[0],points[0]])
    lengths=np.linalg.norm(np.diff(points,axis=0),axis=1)
    knots=np.r_[0.,np.cumsum(lengths)]
    knots=knots/knots[-1] if knots[-1]>1e-8 else np.linspace(0,1,len(points))
    spline=CubicSpline(knots,points,bc_type='natural')
    u_grid=np.linspace(0,1,2001)
    s_grid=10*u_grid**3-15*u_grid**4+6*u_grid**5
    ds=30*u_grid**2-60*u_grid**3+30*u_grid**4
    dds=60*u_grid-180*u_grid**2+120*u_grid**3
    velocity=spline(s_grid,1)*ds[:,None]
    acceleration=spline(s_grid,2)*ds[:,None]**2+spline(s_grid,1)*dds[:,None]
    c=interface['control'];dt=1/c['policy_hz']
    # Fastest uniform time scaling under the sampled path constraints, with 1% margin.
    # Competition deadline is deliberately not an input to motion timing.
    duration=max(.15,1.01*float(np.max(abs(velocity)/np.asarray(c['max_joint_speed_rad_s']))),
        1.01*float(np.sqrt(np.max(abs(acceleration))/max_acceleration)),
        1.875*abs(width_end-width_start)/c['max_gripper_speed_m_s'])
    count=int(np.ceil(duration/dt));u=np.linspace(0,1,count+1)[1:]
    s=10*u**3-15*u**4+6*u**5
    return np.c_[spline(s),width_start+(width_end-width_start)*s]
