"""Estimate tabletop height from measured points, without object ground truth."""
import numpy as np


def tabletop_residual(points, table):
    points = np.asarray(points)
    height = points[..., 2] - table
    pool = points[np.isfinite(points).all(axis=-1) & (abs(height) < .025)]
    if len(pool) <= 300:
        return height, .0005
    pool = pool[::max(1, len(pool)//12000)]
    design = np.c_[pool[:, :2], np.ones(len(pool))]
    inliers = np.ones(len(pool), dtype=bool)
    for _ in range(5):
        coefficients = np.linalg.lstsq(design[inliers], pool[inliers, 2], rcond=None)[0]
        residual = pool[:, 2] - design @ coefficients
        median = np.median(residual)
        sigma = max(.0005, 1.4826*np.median(abs(residual-median)))
        updated = abs(residual-median) < max(.003, 2.5*sigma)
        if updated.sum() < 300:
            break
        inliers = updated
    if np.linalg.norm(coefficients[:2]) >= .10 or abs(coefficients[2]-table) >= .035:
        return height, .0005
    return points[..., 2] - (points[..., 0]*coefficients[0] + points[..., 1]*coefficients[1] + coefficients[2]), sigma
