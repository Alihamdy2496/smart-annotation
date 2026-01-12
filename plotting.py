"""
Plotting functions for visualizing optimization results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from polygon_utils import translate_polygon, get_convex_hull_vertices, unpack_xy


def plot_result(
    xvec, xvec_initial, movables, fixed_obstacles, placement_bounds, search_radius=None
):
    """
    Plot the optimization results showing before and after states in separate files.

    Args:
        xvec: Optimized positions vector
        xvec_initial: Initial positions vector
        movables: List of movable object dictionaries
        fixed_obstacles: List of fixed obstacle dictionaries
        placement_bounds: Tuple of ((xmin, xmax), (ymin, ymax))
        search_radius: KD-tree search radius for visualization (optional)
    """
    pts = unpack_xy(xvec)

    # =============== BEFORE PLOT ===============
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8))
    ax1.set_aspect("equal")
    ax1.set_xlim(placement_bounds[0])
    ax1.set_ylim(placement_bounds[1])
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Before Optimization", fontsize=14, fontweight="bold")

    # plot fixed obstacles
    for idx, obs in enumerate(fixed_obstacles):
        poly = translate_polygon(obs["verts"], obs["center"], obs["RotationAngle"])
        poly_hull = get_convex_hull_vertices(poly, closed=True)
        polygon_patch = Polygon(
            poly_hull[:-1],
            closed=True,
            facecolor="red",
            alpha=0.3,
            edgecolor="darkred",
            linewidth=0.5,
            label="Fixed Obstacles" if idx == 0 else "",
        )
        ax1.add_patch(polygon_patch)
        ax1.plot(poly_hull[:, 0], poly_hull[:, 1], "darkred", linewidth=0.25)

    # plot movables (initial positions)
    if xvec_initial is not None:
        pts_initial = unpack_xy(xvec_initial)
    else:
        pts_initial = pts  # fallback if no initial provided

    for i, p in enumerate(pts_initial):
        poly = translate_polygon(movables[i]["verts"], p, movables[i]["RotationAngle"])
        poly_hull = get_convex_hull_vertices(poly, closed=True)
        polygon_patch = Polygon(
            poly_hull[:-1],
            closed=True,
            facecolor="blue",
            alpha=0.4,
            edgecolor="darkblue",
            linewidth=0.5,
            label="Movable Objects" if i == 0 else "",
        )
        ax1.add_patch(polygon_patch)
        ax1.plot(poly_hull[:, 0], poly_hull[:, 1], "darkblue", linewidth=0.25)
        # Show target point and line from centroid to target
        t = movables[i]["target"]
        hull_verts_calc = get_convex_hull_vertices(poly, closed=False)
        if len(hull_verts_calc) > 0:
            centroid = np.mean(hull_verts_calc, axis=0)
        else:
            centroid = p
        ax1.plot(
            t[0], t[1], "g*", markersize=1, label="Target Points" if i == 0 else ""
        )
        ax1.plot(
            [centroid[0], t[0]], [centroid[1], t[1]], "g--", alpha=0.5, linewidth=0.25
        )

    ax1.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("before.png", dpi=600, bbox_inches="tight")
    plt.close(fig1)
    print("Saved before.png")

    # =============== AFTER PLOT ===============
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
    ax2.set_aspect("equal")
    ax2.set_xlim(placement_bounds[0])
    ax2.set_ylim(placement_bounds[1])
    ax2.grid(True, alpha=0.3)
    ax2.set_title("After Optimization", fontsize=14, fontweight="bold")

    # plot fixed obstacles (same as before)
    for idx, obs in enumerate(fixed_obstacles):
        poly = translate_polygon(obs["verts"], obs["center"], obs["RotationAngle"])
        poly_hull = get_convex_hull_vertices(poly, closed=True)
        polygon_patch = Polygon(
            poly_hull[:-1],
            closed=True,
            facecolor="red",
            alpha=0.3,
            edgecolor="darkred",
            linewidth=0.5,
            label="Fixed Obstacles" if idx == 0 else "",
        )
        ax2.add_patch(polygon_patch)
        ax2.plot(poly_hull[:, 0], poly_hull[:, 1], "darkred", linewidth=0.25)

    # plot movables (optimized positions)
    for i, p in enumerate(pts):
        poly = translate_polygon(movables[i]["verts"], p, movables[i]["RotationAngle"])
        poly_hull = get_convex_hull_vertices(poly, closed=True)
        polygon_patch = Polygon(
            poly_hull[:-1],
            closed=True,
            facecolor="blue",
            alpha=0.4,
            edgecolor="darkblue",
            linewidth=0.5,
            label="Movable Objects" if i == 0 else "",
        )
        ax2.add_patch(polygon_patch)
        ax2.plot(poly_hull[:, 0], poly_hull[:, 1], "darkblue", linewidth=0.25)
        # Show target point and line from centroid to target
        t = movables[i]["target"]
        hull_verts_calc = get_convex_hull_vertices(poly, closed=False)
        if len(hull_verts_calc) > 0:
            centroid = np.mean(hull_verts_calc, axis=0)
        else:
            centroid = p
        ax2.plot(
            t[0], t[1], "g*", markersize=1, label="Target Points" if i == 0 else ""
        )
        ax2.plot(
            [centroid[0], t[0]], [centroid[1], t[1]], "g--", alpha=0.5, linewidth=0.25
        )

    # Visualize KD-tree search radius for a sample of movables
    if search_radius is not None and search_radius > 0:
        # Show radius for up to 5 evenly spaced movables
        num_samples = min(5, len(movables))
        sample_indices = np.linspace(0, len(movables) - 1, num_samples, dtype=int)

        for idx, i in enumerate(sample_indices):
            t = movables[i]["target"]
            circle = plt.Circle(
                t,
                search_radius,
                fill=False,
                edgecolor="orange",
                linestyle="--",
                linewidth=0.5,
                alpha=0.5,
                label="KD-tree Search Radius" if idx == 0 else "",
            )
            ax2.add_patch(circle)

    ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("after.png", dpi=600, bbox_inches="tight")
    plt.close(fig2)
    print("Saved after.png")
