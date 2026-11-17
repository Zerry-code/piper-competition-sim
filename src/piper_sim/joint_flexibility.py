"""Soft full-arm configuration quality, without joint-count or elbow-height targets."""
import numpy as np

def configuration_quality(q,seed,lower,upper,jacobian):
    span=np.maximum(np.asarray(upper)-lower,1e-6)
    normalized=(np.asarray(q)-lower)/span
    margin=np.minimum(normalized,1-normalized)
    singular_values=np.linalg.svd(np.asarray(jacobian)*span[None,:],compute_uv=False)
    sigma=float(singular_values[-1])
    continuous=float(np.max(abs(np.asarray(q)-seed)/span))
    # Every valid joint can participate. These are ranking costs, never exclusions.
    limit_cost=float(np.mean(np.maximum(0.,.15-margin)**2))
    cost=12*limit_cost+.02/(sigma+.01)+.1*continuous**2
    return dict(cost=cost,min_limit_margin=float(np.min(margin)),joint_margins=margin.tolist(),sigma_min=sigma,normalized_max_step=continuous)
