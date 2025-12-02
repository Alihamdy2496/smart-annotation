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
    get movables close to targets
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

    print("objective: ", val)
    return val


def create_non_overlap_constraints(movables, fixed_obstacles, min_separation=0.0):
    """
    Create constraint functions for non-overlap conditions.
    
    Args:
        movables: List of movable object dictionaries
        fixed_obstacles: List of fixed obstacle dictionaries
        min_separation: Minimum required separation distance (default 0.0 for just touching)
    
    Returns:
        List of constraint dictionaries for scipy.optimize
    """
    constraints = []
    
    # Constraint: movable-movable non-overlap
    n_movables = len(movables)
    for i in range(n_movables):
        for j in range(i+1, n_movables):
            # Capture loop variables using default arguments
            def make_constraint_movable(i_val=i, j_val=j):
                def constraint(xvec):
                    pts = unpack_xy(xvec)
                    A_translated = translate_polygon(movables[i_val]['verts'], pts[i_val])
                    B_translated = translate_polygon(movables[j_val]['verts'], pts[j_val])
                    sep = separating_distance(A_translated, B_translated)
                    # Constraint satisfied when sep >= min_separation
                    # Return value must be >= 0 for constraint to be satisfied
                    return sep - min_separation
                return constraint
            
            constraints.append({
                'type': 'ineq',
                'fun': make_constraint_movable(i, j)
            })
    
    # Constraint: movable-fixed non-overlap
    for i in range(n_movables):
        for obs_idx, obs in enumerate(fixed_obstacles):
            # Capture loop variables using default arguments
            def make_constraint_fixed(i_val=i, obs_val=obs):
                def constraint(xvec):
                    pts = unpack_xy(xvec)
                    A_translated = translate_polygon(movables[i_val]['verts'], pts[i_val])
                    B_translated = translate_polygon(obs_val['verts'], obs_val['center'])
                    sep = separating_distance(A_translated, B_translated)
                    # Constraint satisfied when sep >= min_separation
                    return sep - min_separation
                return constraint
            
            constraints.append({
                'type': 'ineq',
                'fun': make_constraint_fixed(i, obs)
            })
    
    return constraints


def optimize(movables, fixed_obstacles, num_restarts=20, maxiter=400, 
             placement_bounds=None, min_separation=0.0):
    """
    Optimize object placement using constraints for non-overlap.
    Uses inequality constraints to enforce hard non-overlap limits.
    
    Args:
        movables: List of movable object dictionaries with 'verts' and 'target' keys
        fixed_obstacles: List of fixed obstacle dictionaries
        num_restarts: Number of random restarts to try
        maxiter: Maximum iterations per optimization
        placement_bounds: Tuple of ((xmin, xmax), (ymin, ymax)) for bounds constraints
        min_separation: Minimum required separation distance between objects (default 0.0)
    
    Returns:
        Tuple of (best result, initial position vector for best result)
    """
    best = None
    best_initial = None
    
    # Set default bounds if not provided
    if placement_bounds is None:
        placement_bounds = ((-10.0, 10.0), (-10.0, 10.0))
    
    # Create bounds for optimizer: list of (min, max) tuples for each variable
    n = len(movables)
    bounds = []
    for _ in range(n):
        bounds.append(placement_bounds[0])  # x bounds: (xmin, xmax)
        bounds.append(placement_bounds[1])  # y bounds: (ymin, ymax)
    
    # Create non-overlap constraints
    constraints = create_non_overlap_constraints(movables, fixed_obstacles, min_separation)
    
    print(f"Created {len(constraints)} non-overlap constraints")
    
    # Create objective function
    def cost_func(x):
        return objective(x, movables)
    
    for restart_idx in range(num_restarts):
        # Better initialization: spread objects around to avoid initial overlaps
        x0 = []
        for i, m in enumerate(movables):
            tx, ty = m['target']
            jitter = 0.3 * (np.random.rand(2) - 0.5)
            x0.append(np.array([tx, ty]) + jitter)
        x0 = np.array(x0).reshape(-1)

        # SLSQP supports constraints, L-BFGS-B does not
        try:
            res = minimize(cost_func, x0, method='SLSQP', bounds=bounds,
                          constraints=constraints,
                          options={'maxiter': maxiter, 'disp': False, 'ftol': 1e-6})
            
            # Check if solution is feasible (all constraints satisfied)
            constraint_violations = []
            for constraint in constraints:
                val = constraint['fun'](res.x)
                constraint_violations.append(val)
            
            min_constraint = min(constraint_violations) if constraint_violations else 0.0
            
            # Only accept feasible solutions
            if min_constraint >= -1e-6:  # Allow small numerical errors
                if best is None or res.fun < best.fun:
                    best = res
                    best_initial = x0.copy()
                    print(f"Restart {restart_idx+1}: Found feasible solution with cost {res.fun:.6f}, "
                          f"min constraint: {min_constraint:.6f}")
                    # If we found a very good solution, break early
                    if res.fun < 1e-3:
                        break
            else:
                print(f"Restart {restart_idx+1}: Infeasible solution (min constraint: {min_constraint:.6f})")
                
        except Exception as e:
            print(f"Restart {restart_idx+1}: Optimization failed: {e}")
            continue
    
    if best is None:
        print("Warning: No feasible solution found. Try increasing num_restarts or adjusting min_separation.")
    
    return best, best_initial

