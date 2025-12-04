"""
Main execution script for object placement optimization.

This script reads problem data (movable objects and fixed obstacles),
runs the optimization, and visualizes the results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from polygon_utils import translate_polygon, get_convex_hull_vertices, unpack_xy
from optimizer import optimize


# ---------------------- Problem data ----------------------
# Movable convex polygons: each has a list of vertices in local coordinates
# and a target point; orientation is fixed so vertices are absolute offsets.
movables = [
    {
        'verts': np.array([[0.5,0.25],[-0.5,0.25],[-0.5,-0.25],[0.5,-0.25]]),
        'target': np.array([0.0,0.25])
    },
    {
        'verts': np.array([[0.25,0.25],[-0.25,0.25],[-0.25,-0.25],[0.25,-0.25]]),
        'target': np.array([1.0,2.25])
    },
    {
        'verts': np.array([[0.25,0.5],[-0.25,0.5],[-0.25,-0.5],[0.25,-0.5]]),
        'target': np.array([2.0,2.5])
    },
    {
        'verts': np.array([[0.5,0.25],[-0.5,0.25],[-0.5,-0.25],[0.5,-0.25]]),
        'target': np.array([0.0,4.75])
    },
]

# Fixed convex polygon obstacles
fixed_obstacles = [
    {
        'verts': np.array([[8.0,0.25],[-8.0,0.25],[8.0,-0.25],[-8.0,-0.25]]),
        'center': np.array([0.0,0.0]),
    },
    {
        'verts': np.array([[0.5,1.5],[-0.5,1.5],[0.5,-1.5],[-0.5,-1.5]]),
        'center': np.array([0.5,2.5]),
    },
    {
        'verts': np.array([[0.5,2.5],[-0.5,2.5],[0.5,-2.5],[-0.5,-2.5]]),
        'center': np.array([2.5,2.5]),
    },
    {
        'verts': np.array([[8.0,0.25],[-8.0,0.25],[8.0,-0.25],[-8.0,-0.25]]),
        'center': np.array([0.0,5.0]),
    },
]

# Optimization parameters
placement_bounds = ((-10.0, 10.0), (-10.0, 10.0))
num_restarts = 10
maxiter = 8000


def plot_result(xvec, xvec_initial, movables, fixed_obstacles, placement_bounds):
    """
    Plot the optimization results showing before and after states.
    
    Args:
        xvec: Optimized positions vector
        xvec_initial: Initial positions vector
        movables: List of movable object dictionaries
        fixed_obstacles: List of fixed obstacle dictionaries
        placement_bounds: Tuple of ((xmin, xmax), (ymin, ymax))
    """
    pts = unpack_xy(xvec)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax in [ax1, ax2]:
        ax.set_aspect('equal')
        ax.set_xlim(placement_bounds[0])
        ax.set_ylim(placement_bounds[1])
        ax.grid(True, alpha=0.3)
    
    # Plot BEFORE optimization
    ax1.set_title('Before', fontsize=14, fontweight='bold')
    
    # plot fixed obstacles
    for idx, obs in enumerate(fixed_obstacles):
        poly = translate_polygon(obs['verts'], obs['center'])
        # Get convex hull vertices for proper drawing
        poly_hull = get_convex_hull_vertices(poly, closed=True)
        # Use matplotlib Polygon for proper rendering
        polygon_patch = Polygon(poly_hull[:-1], closed=True, facecolor='red', 
                               alpha=0.3, edgecolor='darkred', linewidth=2,
                               label='Fixed Obstacles' if idx == 0 else '')
        ax1.add_patch(polygon_patch)
        ax1.plot(poly_hull[:,0], poly_hull[:,1], 'darkred', linewidth=1.5)
    
    # plot movables (initial positions)
    if xvec_initial is not None:
        pts_initial = unpack_xy(xvec_initial)
    else:
        pts_initial = pts  # fallback if no initial provided
    
    for i, p in enumerate(pts_initial):
        poly = translate_polygon(movables[i]['verts'], p)
        # Get convex hull vertices for proper drawing
        poly_hull = get_convex_hull_vertices(poly, closed=True)
        # Use matplotlib Polygon for proper rendering
        polygon_patch = Polygon(poly_hull[:-1], closed=True, facecolor='blue', 
                               alpha=0.4, edgecolor='darkblue', linewidth=2,
                               label='Movable Objects' if i == 0 else '')
        ax1.add_patch(polygon_patch)
        ax1.plot(poly_hull[:,0], poly_hull[:,1], 'darkblue', linewidth=1.5)
        # Show target point and line from centroid to target
        t = movables[i]['target']
        # Calculate centroid of convex hull
        hull_verts_calc = get_convex_hull_vertices(poly, closed=False)
        if len(hull_verts_calc) > 0:
            centroid = np.mean(hull_verts_calc, axis=0)
        else:
            centroid = p
        ax1.plot(t[0], t[1], 'g*', markersize=15, label='Target Points' if i == 0 else '')
        ax1.plot([centroid[0],t[0]],[centroid[1],t[1]],'g--', alpha=0.5, linewidth=1.5)
    
    ax1.legend(loc='upper right')
    
    # Plot AFTER optimization
    ax2.set_title('After', fontsize=14, fontweight='bold')
    
    # plot fixed obstacles (same as before)
    for obs in fixed_obstacles:
        poly = translate_polygon(obs['verts'], obs['center'])
        # Get convex hull vertices for proper drawing
        poly_hull = get_convex_hull_vertices(poly, closed=True)
        # Use matplotlib Polygon for proper rendering
        polygon_patch = Polygon(poly_hull[:-1], closed=True, facecolor='red', 
                               alpha=0.3, edgecolor='darkred', linewidth=2)
        ax2.add_patch(polygon_patch)
        ax2.plot(poly_hull[:,0], poly_hull[:,1], 'darkred', linewidth=1.5)
    
    # plot movables (optimized positions)
    for i, p in enumerate(pts):
        poly = translate_polygon(movables[i]['verts'], p)
        # Get convex hull vertices for proper drawing
        poly_hull = get_convex_hull_vertices(poly, closed=True)
        # Use matplotlib Polygon for proper rendering
        polygon_patch = Polygon(poly_hull[:-1], closed=True, facecolor='blue', 
                               alpha=0.4, edgecolor='darkblue', linewidth=2)
        ax2.add_patch(polygon_patch)
        ax2.plot(poly_hull[:,0], poly_hull[:,1], 'darkblue', linewidth=1.5)
        # Show target point and line from centroid to target
        t = movables[i]['target']
        # Calculate centroid of convex hull
        hull_verts_calc = get_convex_hull_vertices(poly, closed=False)
        if len(hull_verts_calc) > 0:
            centroid = np.mean(hull_verts_calc, axis=0)
        else:
            centroid = p
        ax2.plot(t[0], t[1], 'g*', markersize=15)
        ax2.plot([centroid[0],t[0]],[centroid[1],t[1]],'g--', alpha=0.5, linewidth=1.5)
    
    plt.tight_layout()
    plt.savefig('result.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    np.random.seed(0)
    
    # Run optimization with adaptive margins
    # margin_ratio: 0.1 = 10% of average polygon size as margin
    # min_separation: base separation distance (added to adaptive margin)
    # target_weight: weight for distance to target (default 100.0)
    # normal_weight: weight for alignment with normal from fixed obstacles (default 10.0)
    res, x_initial = optimize(movables, fixed_obstacles, num_restarts, maxiter, 
                              placement_bounds, min_separation=0.0, margin_ratio=0.1,
                              target_weight=100.0, normal_weight=100.0)
    
    print("Result:", res.fun)
    
    # Visualize results
    plot_result(res.x, x_initial, movables, fixed_obstacles, placement_bounds)

