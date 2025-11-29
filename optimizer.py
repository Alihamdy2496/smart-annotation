"""
Optimizer and cost functions for object placement optimization.
"""

import numpy as np
from scipy.optimize import minimize
from polygon_utils import (
    unpack_xy, translate_polygon, get_convex_hull_vertices, separating_distance
)


def objective(xvec, movables):
    """
    Calculate objective: sum of squared distances from convex hull centroids to targets.
    
    Args:
        xvec: Flattened array of positions for all movable objects
        movables: List of movable object dictionaries with 'verts' and 'target' keys
    
    Returns:
        Total squared distance from centroids to targets
    """
    pts = unpack_xy(xvec)
    val = 0.0
    for i, p in enumerate(pts):
        # Get the translated polygon
        poly = translate_polygon(movables[i]['verts'], p)
        # Get convex hull vertices
        hull_verts = get_convex_hull_vertices(poly, closed=False)
        # Calculate centroid of convex hull
        if len(hull_verts) > 0:
            centroid = np.mean(hull_verts, axis=0)
        else:
            centroid = p
        # Distance from centroid to target
        t = movables[i]['target']
        val += np.sum((centroid - t)**2)
    return val


def overlap_penalty(xvec, movables, fixed_obstacles, weight=1000.0):
    """
    Calculate overlap penalty using convex hull vertices for SAT collision detection.
    
    Args:
        xvec: Flattened array of positions for all movable objects
        movables: List of movable object dictionaries
        fixed_obstacles: List of fixed obstacle dictionaries with 'verts' and 'center' keys
        weight: Penalty weight multiplier
    
    Returns:
        Total overlap penalty
    """
    pts = unpack_xy(xvec)
    penalty = 0.0
    # movable-movable
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            # Get translated polygons (pass original vertices, let separating_distance handle convex hull)
            A_translated = translate_polygon(movables[i]['verts'], pts[i])
            B_translated = translate_polygon(movables[j]['verts'], pts[j])
            # separating_distance will compute convex hull internally via polygon_edges
            sep = separating_distance(A_translated, B_translated)
            if sep < 0:
                penalty += weight * sep**2
    # movable-fixed
    for i in range(len(pts)):
        for obs in fixed_obstacles:
            # Get translated polygons (pass original vertices, let separating_distance handle convex hull)
            B_translated = translate_polygon(obs['verts'], obs['center'])
            A_translated = translate_polygon(movables[i]['verts'], pts[i])
            # separating_distance will compute convex hull internally via polygon_edges
            sep = separating_distance(A_translated, B_translated)
            if sep < 0:
                penalty += weight * sep**2

    return penalty


def total_cost(xvec, movables, fixed_obstacles):
    """
    Total cost function combining objective and overlap penalty.
    
    Args:
        xvec: Flattened array of positions for all movable objects
        movables: List of movable object dictionaries
        fixed_obstacles: List of fixed obstacle dictionaries
    
    Returns:
        Total cost
    """
    return objective(xvec, movables) + overlap_penalty(xvec, movables, fixed_obstacles)


def optimize(movables, fixed_obstacles, num_restarts=20, maxiter=400):
    """
    Optimize object placement using multiple random restarts.
    
    Args:
        movables: List of movable object dictionaries with 'verts' and 'target' keys
        fixed_obstacles: List of fixed obstacle dictionaries
        num_restarts: Number of random restarts to try
        maxiter: Maximum iterations per optimization
    
    Returns:
        Tuple of (best result, initial position vector for best result)
    """
    best = None
    best_initial = None
    
    for _ in range(num_restarts):
        x0 = []
        for m in movables:
            tx, ty = m['target']
            jitter = 0.5 * (np.random.rand(2) - 0.5)
            x0.append(np.array([tx, ty]) + jitter)
        x0 = np.array(x0).reshape(-1)

        # Create cost function with current problem data
        def cost_func(x):
            return total_cost(x, movables, fixed_obstacles)

        res = minimize(cost_func, x0, method='SLSQP',
                      options={'maxiter': maxiter, 'disp': False})

        if best is None or res.fun < best.fun:
            best = res
            best_initial = x0.copy()
    
    return best, best_initial

