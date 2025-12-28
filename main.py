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


import json

# ---------------------- Problem data ----------------------
# Load data from JSON
json_path = "AnnotationCleaner_CurveLoops.json"
with open(json_path, "r") as f:
    data = json.load(f)

movables = []
fixed_obstacles = []

for item in data:
    # Extract vertices (X, Y only for 2D optimization)
    # Note: The user requested "Vertices from the file center which is the origin" for fixed_obstacles
    # and "Vertices from the file, target which is the origin" for movables.
    # Looking at the original code:
    # Movables had 'verts' (relative to some center?) and 'target' (absolute position).
    # Fixed obstacles had 'verts' (relative to center) and 'center' (absolute position).

    # The user request says:
    # fixed_obstacles has verts which is Vertices from the file center which is the origin and add the ElementId and RotationAngle
    # movables list has verts which is Vertices from the file, target which is the origin and add the ElementId and RotationAngle

    # Interpreting "center which is the origin" and "target which is the origin":
    # It seems the user wants the 'Origin' from JSON to be the 'center' for fixed obstacles and 'target' for movables.
    # And 'Vertices' from JSON are likely absolute coordinates?
    # If 'Vertices' are absolute, we might need to make them relative to the origin if the optimizer expects relative verts.
    # However, the user said "verts which is Vertices from the file", implying direct usage.
    # BUT, the original code for fixed obstacles had relative verts and a center.
    # Let's look at the JSON data again.
    # Example Fixed: Origin X=167.88..., Vertices X=167.88...
    # It seems Vertices are absolute.
    # If the optimizer expects relative vertices (which is typical for "placing" objects), we should subtract Origin from Vertices.
    # The user phrasing "verts which is Vertices from the file center which is the origin" is slightly ambiguous.
    # It could mean: Use Vertices from file. Set center = Origin.
    # OR: Vertices should be relative to center (which is the origin of the local frame).

    # Let's assume standard rigid body physics/optimization where shape is defined in local frame (relative to center)
    # and center/target is the position in world frame.
    # So: LocalVerts = AbsoluteVerts - Origin.

    # Re-reading user request carefully:
    # "fixed_obstacles has verts which is Vertices from the file center which is the origin"
    # "movables list has verts which is Vertices from the file, target which is the origin"

    # This likely means:
    # fixed_obstacles:
    #   verts = [v - origin for v in Vertices]  <-- This makes them local
    #   center = Origin
    # movables:
    #   verts = [v - origin for v in Vertices]  <-- This makes them local
    #   target = Origin

    origin = np.array([item["Origin"]["X"], item["Origin"]["Y"]])
    verts_absolute = np.array([[v["X"], v["Y"]] for v in item["Vertices"]])
    verts_local = verts_absolute - origin

    element = {
        "verts": verts_local,
        "ElementId": item["ElementId"],
        "HostElementId": item["HostElementId"],
        "RotationAngle": item["RotationAngle"],
    }

    if item.get("IsMovable"):
        element["target"] = origin
        movables.append(element)
    elif item.get("IsFixed"):
        element["center"] = origin
        fixed_obstacles.append(element)

# Optimization parameters
# Compute placement bounds from the data
all_coords = []
for item in data:
    for v in item["Vertices"]:
        all_coords.append([v["X"], v["Y"]])

all_coords = np.array(all_coords)
x_min, y_min = all_coords.min(axis=0)
x_max, y_max = all_coords.max(axis=0)

# Add margin (20% of the range on each side)
x_range = x_max - x_min
y_range = y_max - y_min
margin_x = 0.2 * x_range
margin_y = 0.2 * y_range

placement_bounds = (
    (x_min - margin_x, x_max + margin_x),
    (y_min - margin_y, y_max + margin_y),
)

print(f"Computed placement bounds: X={placement_bounds[0]}, Y={placement_bounds[1]}")

num_restarts = 1
maxiter = 8000


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


if __name__ == "__main__":
    np.random.seed(0)

    # Run optimization with adaptive margins
    # margin_ratio: 0.1 = 10% of average polygon size as margin
    # min_separation: base separation distance (added to adaptive margin)
    # target_weight: weight for distance to target (default 100.0)
    # normal_weight: weight for alignment with normal from fixed obstacles (default 10.0)
    result, x0, search_radius = optimize(
        movables,
        fixed_obstacles,
        num_restarts,
        maxiter,
        placement_bounds,
        min_separation=0.5,
        margin_ratio=0.1,
        target_weight=100.0,
        normal_weight=100.0,
    )

    # Visualize results
    plot_result(result, x0, movables, fixed_obstacles, placement_bounds, search_radius)
