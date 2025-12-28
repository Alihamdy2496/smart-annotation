"""
Optimizer and cost functions for object placement optimization.
"""

import numpy as np
from scipy.optimize import minimize, BFGS
from scipy.spatial import cKDTree
from joblib import Parallel, delayed
from polygon_utils import (
    unpack_xy,
    translate_polygon,
    get_convex_hull_vertices,
    separating_distance,
    separating_distance_SAT,
    polygon_characteristic_size,
    get_precomputed_geometry,
    separating_distance_SAT_precomputed,
)
from math import exp


def overlap_penalty(
    xvec,
    movables,
    all_fixed_obstacles,
    overlap_weight=1000.0,
    min_separation=0.2,
    margin_ratio=0.1,
    movable_sizes=None,
    fixed_sizes=None,
):
    """
    Calculate overlap penalty with KD-tree spatial filtering.

    Args:
        xvec: Flattened array of positions for all movable objects
        movables: List of movable object dictionaries
        all_fixed_obstacles: List of all fixed obstacle dictionaries
        overlap_weight: Weight for overlap penalty term
        min_separation: Base minimum separation distance
        margin_ratio: Ratio of average polygon size to use as margin
        movable_sizes: Pre-computed sizes of movables
        fixed_sizes: Pre-computed sizes of fixed obstacles

    Returns:
        Overlap penalty value
    """
    pts = unpack_xy(xvec)
    val = 0.0

    # Calculate search radius based on maximum object sizes
    n_movables = len(movables)
    if n_movables > 0 and movable_sizes is not None and fixed_sizes is not None:
        max_movable_size = np.max(movable_sizes)
        max_fixed_size = (
            np.max(
                [
                    fixed_sizes[idx]
                    for idx, obs in enumerate(all_fixed_obstacles)
                    if obs.get("center") is not None
                ]
            )
            if any(obs.get("center") is not None for obs in all_fixed_obstacles)
            else 1.0
        )
        search_radius = (max_movable_size) * 4.0

        # Build KD-tree for current movable positions
        movable_tree = cKDTree(pts)

        # Penalize overlaps between movables (spatially filtered)
        for i in range(n_movables):
            # Query KD-tree to find nearby movables within search radius
            candidate_indices = movable_tree.query_ball_point(pts[i], search_radius)

            for j in candidate_indices:
                if j <= i:  # Skip self and already processed pairs
                    continue

                # Calculate adaptive margin based on polygon sizes
                size_i = movable_sizes[i]
                size_j = movable_sizes[j]
                avg_size = (size_i + size_j) / 2.0
                adaptive_margin = min_separation + (
                    margin_ratio * avg_size if margin_ratio is not None else 0.0
                )

                A_translated = translate_polygon(
                    movables[i]["verts"],
                    pts[i],
                    movables[i]["RotationAngle"],
                )
                B_translated = translate_polygon(
                    movables[j]["verts"],
                    pts[j],
                    movables[j]["RotationAngle"],
                )
                sep = separating_distance(A_translated, B_translated)

                # Penalize if separation < margin
                # The penalty increases quadratically as overlap increases
                if sep < adaptive_margin:
                    # violation = adaptive_margin - sep
                    # val += overlap_weight * violation**2
                    val += overlap_weight * exp(-(sep / 0.1))
        # Build KD-tree for fixed obstacle centers
        fixed_centers = np.array(
            [
                obs["center"]
                for obs in all_fixed_obstacles
                if obs.get("center") is not None
            ]
        )
        valid_fixed_indices = [
            idx
            for idx, obs in enumerate(all_fixed_obstacles)
            if obs.get("center") is not None
        ]

        if len(fixed_centers) > 0:
            fixed_tree = cKDTree(fixed_centers)

            # Penalize overlaps between movables and fixed obstacles (spatially filtered)
            for i in range(n_movables):
                # Query KD-tree to find nearby fixed obstacles within search radius
                candidate_tree_indices = fixed_tree.query_ball_point(
                    pts[i], search_radius
                )

                for tree_idx in candidate_tree_indices:
                    obs_idx = valid_fixed_indices[tree_idx]
                    obs = all_fixed_obstacles[obs_idx]

                    # Calculate adaptive margin based on polygon sizes
                    size_i = movable_sizes[i]
                    size_obs = fixed_sizes[obs_idx]
                    avg_size = (size_i + size_obs) / 2.0
                    adaptive_margin = min_separation + (
                        margin_ratio * avg_size if margin_ratio is not None else 0.0
                    )

                    A_translated = translate_polygon(
                        movables[i]["verts"],
                        pts[i],
                        movables[i]["RotationAngle"],
                    )
                    B_translated = translate_polygon(
                        obs["verts"],
                        obs["center"],
                        obs["RotationAngle"],
                    )
                    sep = separating_distance(A_translated, B_translated)

                    # Penalize if separation < margin
                    if sep < adaptive_margin:
                        # violation = adaptive_margin - sep
                        # val += overlap_weight * violation**2
                        val += overlap_weight * exp(-(sep / 0.1))

    return val


def objective(
    xvec,
    movables,
    fixed_obstacles,
    all_fixed_obstacles,
    target_weight=100.0,
    normal_weight=100.0,
    overlap_weight=1000.0,
    min_separation=0.2,
    margin_ratio=0.1,
    movable_sizes=None,
    fixed_sizes=None,
):
    """
    Calculate objective with three components:
    1. Distance from movables to their targets
    2. Distance from movables to normal lines from fixed obstacles
    3. Penalty for overlaps between movables and between movables and fixed obstacles

    The normal is computed from the line connecting fixed_obstacle center to movable target.

    Args:
        xvec: Flattened array of positions for all movable objects
        movables: List of movable object dictionaries with 'verts' and 'target' keys
        fixed_obstacles: List of fixed obstacle dictionaries with 'center' key
        target_weight: Weight for target distance term
        normal_weight: Weight for normal alignment term
        overlap_weight: Weight for overlap penalty term
        min_separation: Base minimum separation distance
        margin_ratio: Ratio of average polygon size to use as margin

    Returns:
        Total weighted objective value
    """
    pts = unpack_xy(xvec)
    val = 0.0

    for i, p in enumerate(pts):
        t = movables[i]["target"]
        center = fixed_obstacles[i]["center"]
        if center is None:
            continue

        # Component 1: Distance from movables to targets
        val += target_weight * np.sum((p - t) ** 2)

        # Component 2: Distance from movables to normal lines from fixed obstacles
        # Vector from fixed obstacle center to movable target
        vec_to_target = t - center
        vec_len = np.linalg.norm(vec_to_target)

        # Skip if target is at the center (degenerate case)
        if vec_len < 1e-10:
            continue

        # Normalize the vector
        vec_to_target_norm = vec_to_target / vec_len

        # Compute normal vector (perpendicular to vec_to_target)
        # In 2D, normal to [dx, dy] is [-dy, dx] or [dy, -dx]
        normal = np.array([-vec_to_target_norm[1], vec_to_target_norm[0]])

        # Vector from fixed obstacle center to movable position
        vec_to_movable = p - center

        # Project vec_to_movable onto the normal direction
        # Distance from movable to the normal line through fixed center
        # This is the component of vec_to_movable along the normal
        distance_to_normal = np.abs(np.dot(vec_to_movable, normal))

        # Add squared distance to objective (minimize distance to normal line)
        val += normal_weight * distance_to_normal**2

    return val


def find_all_overlaps(x, movables, fixed_obstacles, min_separation=0.0):
    """
    Find all overlapping pairs between movables and between movables and fixed obstacles.
    Optimized with precomputed geometry and parallelization.
    """
    pts = unpack_xy(x)
    overlaps = []

    # Ensure precomputed geometry exists
    for m in movables:
        if "precomputed" not in m:
            m["precomputed"] = get_precomputed_geometry(
                m["verts"], m.get("RotationAngle", 0.0)
            )
    for obs in fixed_obstacles:
        if obs.get("center") is not None and "precomputed" not in obs:
            obs["precomputed"] = get_precomputed_geometry(
                obs["verts"], obs.get("RotationAngle", 0.0)
            )

    n_movables = len(movables)

    # Helper for movable-movable check
    def check_movable_movable(i, j, min_sep):
        hullA = movables[i]["precomputed"]["hull"] + pts[i]
        hullB = movables[j]["precomputed"]["hull"] + pts[j]

        sep, normal, penetration = separating_distance_SAT_precomputed(
            hullA,
            hullB,
            movables[i]["precomputed"]["normals"],
            movables[j]["precomputed"]["normals"],
            return_normal=True,
        )

        if sep < min_sep - 1e-6:
            return (i, j, min_sep - sep, normal)
        return None

    # Helper for movable-fixed check
    def check_movable_fixed(i, obs_idx, min_sep):
        obs = fixed_obstacles[obs_idx]
        if obs.get("center") is None:
            return None

        hullA = movables[i]["precomputed"]["hull"] + pts[i]
        hullB = obs["precomputed"]["hull"] + obs["center"]

        sep, normal, penetration = separating_distance_SAT_precomputed(
            hullA,
            hullB,
            movables[i]["precomputed"]["normals"],
            obs["precomputed"]["normals"],
            return_normal=True,
        )

        if sep < min_sep - 1e-6:
            return (i, None, min_sep - sep, normal)
        return None

    # Parallelize movable-movable checks
    if n_movables > 1:
        num_mm_pairs = n_movables * (n_movables - 1) // 2
        if num_mm_pairs > 50:
            mm_results = Parallel(n_jobs=-1, backend="threading")(
                delayed(check_movable_movable)(i, j, min_separation)
                for i in range(n_movables)
                for j in range(i + 1, n_movables)
            )
            overlaps.extend([r for r in mm_results if r is not None])
        else:
            for i in range(n_movables):
                for j in range(i + 1, n_movables):
                    res = check_movable_movable(i, j, min_separation)
                    if res:
                        overlaps.append(res)

    # Parallelize movable-fixed checks
    if n_movables > 0 and len(fixed_obstacles) > 0:
        num_mf_pairs = n_movables * len(fixed_obstacles)
        if num_mf_pairs > 50:
            mf_results = Parallel(n_jobs=-1, backend="threading")(
                delayed(check_movable_fixed)(i, obs_idx, min_separation)
                for i in range(n_movables)
                for obs_idx in range(len(fixed_obstacles))
            )
            overlaps.extend([r for r in mf_results if r is not None])
        else:
            for i in range(n_movables):
                for obs_idx in range(len(fixed_obstacles)):
                    res = check_movable_fixed(i, obs_idx, min_separation)
                    if res:
                        overlaps.append(res)

    return overlaps


def project_to_nonoverlap(
    x, movables, fixed_obstacles, max_proj_iters=50, min_separation=0.2
):
    """
    Project positions to enforce non-overlap condition using geometry-based projection.

    This function iteratively:
    1. Finds all overlapping pairs
    2. Resolves the worst overlap (largest penetration)
    3. Recalculates overlaps
    4. Repeats until no overlaps remain or max iterations reached

    This is deterministic, geometry-based, and fast.

    Args:
        x: Flattened array of positions for all movable objects
        movables: List of movable object dictionaries
        fixed_obstacles: List of fixed obstacle dictionaries
        max_proj_iters: Maximum number of projection iterations (default: 50)
        min_separation: Minimum separation distance between objects (default: 0.0)
                       If > 0, objects will maintain a gap instead of just touching

    Returns:
        Projected positions (flattened array) with overlaps resolved
    """
    pts = unpack_xy(x).copy()

    for iteration in range(max_proj_iters):
        overlaps = find_all_overlaps(
            pts.reshape(-1), movables, fixed_obstacles, min_separation
        )

        if not overlaps:
            # No overlaps found, we're done
            break

        # Resolve the worst overlap first (largest penetration)
        # This helps with convergence when there are multiple overlaps
        overlaps.sort(key=lambda o: o[2], reverse=True)
        i, j, separation_distance, normal = overlaps[0]

        if j is None:
            # Fixed obstacle: only move the movable object
            pts[i] += separation_distance * normal
        else:
            # Two movables: move both equally
            pts[i] += 0.5 * separation_distance * normal
            pts[j] -= 0.5 * separation_distance * normal

    return pts.reshape(-1)


def create_non_overlap_constraints(
    movables, fixed_obstacles, min_separation=0.0, margin_ratio=0.1
):
    """
    Create constraint functions for non-overlap conditions with adaptive margins.
    Uses KD-tree spatial filtering to only create constraints for nearby obstacles.

    The margin is calculated as: min_separation + margin_ratio * average_polygon_size
    This ensures larger polygons get proportionally larger margins.

    Args:
        movables: List of movable object dictionaries
        fixed_obstacles: List of fixed obstacle dictionaries
        min_separation: Base minimum separation distance (added to adaptive margin)
                       Default: 0.0 (just touching)
        margin_ratio: Ratio of average polygon size to use as margin
                      - 0.0 = no adaptive margin (only min_separation)
                      - 0.05 = 5% of average polygon size
                      - 0.1 = 10% of average polygon size (recommended)
                      - 0.2 = 20% of average polygon size (larger safety margin)
                      If None, uses only min_separation

    Returns:
        List of constraint dictionaries for scipy.optimize
    """
    constraints = []

    # Pre-compute polygon sizes for adaptive margins
    movable_sizes = [polygon_characteristic_size(m["verts"]) for m in movables]
    fixed_sizes = [polygon_characteristic_size(obs["verts"]) for obs in fixed_obstacles]

    # Calculate search radius based on maximum object sizes
    max_movable_size = np.max(movable_sizes) if len(movable_sizes) > 0 else 1.0
    max_fixed_size = (
        np.max(
            [
                fixed_sizes[idx]
                for idx, obs in enumerate(fixed_obstacles)
                if obs.get("center") is not None
            ]
        )
        if any(obs.get("center") is not None for obs in fixed_obstacles)
        else 1.0
    )
    search_radius = (max_movable_size) * 4.0  # 4x safety factor

    # Build KD-tree for movable targets
    n_movables = len(movables)
    movable_targets = np.array([m["target"] for m in movables])
    movable_tree = cKDTree(movable_targets)

    # Constraint: movable-movable non-overlap (spatially filtered)
    total_movable_pairs = 0
    for i in range(n_movables):
        # Query KD-tree to find nearby movables within search radius
        # query_ball_point returns indices of points within radius
        candidate_indices = movable_tree.query_ball_point(
            movables[i]["target"], search_radius
        )

        # Only create constraints with movables that have index > i to avoid duplicates
        for j in candidate_indices:
            if j <= i:  # Skip self and already processed pairs
                continue

            total_movable_pairs += 1

            # Calculate adaptive margin based on polygon sizes
            size_i = movable_sizes[i]
            size_j = movable_sizes[j]
            avg_size = (size_i + size_j) / 2.0
            adaptive_margin = min_separation + (
                margin_ratio * avg_size if margin_ratio is not None else 0.0
            )

            # Capture loop variables using default arguments
            def make_constraint_movable(i_val=i, j_val=j, margin=adaptive_margin):
                def constraint(xvec):
                    pts = unpack_xy(xvec)
                    A_translated = translate_polygon(
                        movables[i_val]["verts"],
                        pts[i_val],
                        movables[i_val]["RotationAngle"],
                    )
                    B_translated = translate_polygon(
                        movables[j_val]["verts"],
                        pts[j_val],
                        movables[j_val]["RotationAngle"],
                    )
                    sep = separating_distance(A_translated, B_translated)
                    # Constraint satisfied when sep >= margin
                    # Return value must be >= 0 for constraint to be satisfied
                    return sep - margin

                return constraint

            constraints.append(
                {"type": "ineq", "fun": make_constraint_movable(i, j, adaptive_margin)}
            )

    max_movable_pairs = n_movables * (n_movables - 1) // 2
    print(
        f"KD-tree filtering reduced movable-movable pairs from {max_movable_pairs} to {total_movable_pairs}"
    )

    # Build KD-tree for spatial filtering of fixed obstacles
    # Only create constraints for movables near fixed obstacles
    fixed_centers = np.array(
        [obs["center"] for obs in fixed_obstacles if obs.get("center") is not None]
    )
    valid_fixed_indices = [
        idx for idx, obs in enumerate(fixed_obstacles) if obs.get("center") is not None
    ]

    if len(fixed_centers) > 0:
        tree = cKDTree(fixed_centers)

        # Calculate search radius based on maximum movable size
        max_movable_size = np.max(movable_sizes) if len(movable_sizes) > 0 else 1.0
        max_fixed_size = (
            np.max([fixed_sizes[idx] for idx in valid_fixed_indices])
            if len(valid_fixed_indices) > 0
            else 1.0
        )
        search_radius = (max_movable_size) * 2  # 2x safety factor

        # Constraint: movable-fixed non-overlap (spatially filtered)
        total_candidates = 0
        for i in range(n_movables):
            # Query KD-tree to find fixed obstacles within search radius of movable's target
            candidate_tree_indices = tree.query_ball_point(
                movables[i]["target"], search_radius
            )
            total_candidates += len(candidate_tree_indices)

            for tree_idx in candidate_tree_indices:
                # Map tree index back to original fixed_obstacles index
                obs_idx = valid_fixed_indices[tree_idx]
                obs = fixed_obstacles[obs_idx]

                # Calculate adaptive margin based on polygon sizes
                size_i = movable_sizes[i]
                size_obs = fixed_sizes[obs_idx]
                avg_size = (size_i + size_obs) / 2.0
                adaptive_margin = min_separation + (
                    margin_ratio * avg_size if margin_ratio is not None else 0.0
                )

                # Capture loop variables using default arguments
                def make_constraint_fixed(i_val=i, obs_val=obs, margin=adaptive_margin):
                    def constraint(xvec):
                        pts = unpack_xy(xvec)
                        A_translated = translate_polygon(
                            movables[i_val]["verts"],
                            pts[i_val],
                            movables[i_val]["RotationAngle"],
                        )
                        B_translated = translate_polygon(
                            obs_val["verts"],
                            obs_val["center"],
                            obs_val["RotationAngle"],
                        )
                        sep = separating_distance(A_translated, B_translated)
                        # Constraint satisfied when sep >= margin
                        return sep - margin

                    return constraint

                constraints.append(
                    {
                        "type": "ineq",
                        "fun": make_constraint_fixed(i, obs, adaptive_margin),
                    }
                )

        print(
            f"KD-tree filtering reduced movable-fixed pairs from {n_movables * len(fixed_obstacles)} to {total_candidates}"
        )
    else:
        print(
            "No valid fixed obstacles with centers, skipping movable-fixed constraints"
        )

    return constraints


def create_positive_side_constraints(movables, fixed_obstacles):
    """
    Create constraints to ensure movables are on the positive side (in front) of fixed obstacles.

    For each movable and fixed obstacle pair:
    - Compute vector from fixed_obstacle center to movable target
    - Compute normal vector (perpendicular) to this vector
    - Ensure movable position is on the positive side of this normal (dot product >= 0)

    Args:
        movables: List of movable object dictionaries with 'target' key
        fixed_obstacles: List of fixed obstacle dictionaries with 'center' key

    Returns:
        List of constraint dictionaries for scipy.optimize
    """
    constraints = []
    n_movables = len(movables)

    for i in range(n_movables):
        t = movables[i]["target"]

        for obs_idx, obs in enumerate(fixed_obstacles):
            center = obs["center"]

            # Vector from fixed obstacle center to movable target
            vec_to_target = t - center
            vec_len = np.linalg.norm(vec_to_target)

            # Skip if target is at the center (degenerate case)
            if vec_len < 1e-10:
                vec_len = 1e-10

            # Normalize the vector
            normal = vec_to_target / vec_len

            # Compute normal vector (perpendicular to vec_to_target)
            # In 2D, normal to [dx, dy] is [-dy, dx] (90 degree counter-clockwise rotation)
            # This normal points to the "positive side" (in front of the obstacle)
            # normal = np.array([-vec_to_target_norm[1], vec_to_target_norm[0]])

            # Capture loop variables using default arguments (copy normal to avoid reference issues)
            normal_copy = normal.copy()
            center_copy = center.copy()

            def make_positive_side_constraint(
                i_val=i, obs_center=center_copy, normal_vec=normal_copy
            ):
                def constraint(xvec):
                    pts = unpack_xy(xvec)
                    p = pts[i_val]

                    # Vector from fixed obstacle center to movable position
                    vec_to_movable = p - obs_center
                    vec_len = np.linalg.norm(vec_to_movable)

                    # Skip if target is at the center (degenerate case)
                    if vec_len < 1e-10:
                        vec_len = 1e-10

                    # Normalize the vector
                    vec_to_movable_norm = vec_to_movable / vec_len

                    # Dot product: positive means movable is on positive side of normal
                    # Constraint satisfied when dot product >= 0
                    return np.dot(vec_to_movable_norm, normal_vec)

                return constraint

            constraints.append(
                {
                    "type": "ineq",
                    "fun": make_positive_side_constraint(i, center_copy, normal_copy),
                }
            )

    return constraints


def generate_initial_point(movables, fixed_obstacles, init_attempt, prev_solution=None):
    def project_to_positive_side(p, center, normal, min_dist=0.05):
        """
        Ensure p is on the positive half-plane defined by 'normal' through 'center'.
        """
        v = p - center
        d = np.dot(v, normal)

        if d >= min_dist:
            return p  # already OK

        return center + normal * min_dist

    def resolve_initial_collisions(points, movables, fixed_obstacles, iterations=10):
        pts = points.copy()

        for _ in range(iterations):
            # Movable vs movable
            for i in range(len(movables)):
                for j in range(i + 1, len(movables)):
                    A = translate_polygon(
                        movables[i]["verts"],
                        pts[i],
                        movables[i]["RotationAngle"],
                    )
                    B = translate_polygon(
                        movables[j]["verts"],
                        pts[j],
                        movables[j]["RotationAngle"],
                    )
                    sep = separating_distance(A, B)
                    if sep < 0:  # overlapping
                        direction = pts[i] - pts[j]
                        if np.linalg.norm(direction) < 1e-6:
                            direction = np.random.randn(2)
                        direction = direction / np.linalg.norm(direction)
                        pts[i] += direction * (-sep * 0.5)
                        pts[j] -= direction * (-sep * 0.5)

            # Movable vs fixed
            for i in range(len(movables)):
                for obs in fixed_obstacles:
                    A = translate_polygon(
                        movables[i]["verts"],
                        pts[i],
                        movables[i]["RotationAngle"],
                    )
                    B = translate_polygon(
                        obs["verts"], obs["center"], obs["RotationAngle"]
                    )
                    sep = separating_distance(A, B)
                    ################################### here make usre that this doesn't violate the positive side
                    if sep < 0:
                        direction = pts[i] - obs["center"]
                        if np.linalg.norm(direction) < 1e-6:
                            direction = np.random.randn(2)
                        direction = direction / np.linalg.norm(direction)
                        pts[i] += direction * (-sep)

        return pts

    pts = []

    # Step A: start at target or previous good solution
    for i, m in enumerate(movables):
        if prev_solution is not None:
            p = prev_solution[i].copy()
        else:
            p = m["target"].copy()

        # Step B: jitter if previous attempt OR if p violates constraints
        if init_attempt > 0:
            p += 0.5 * (np.random.rand(2) - 0.5)

        pts.append(p)

    pts = np.array(pts)

    # Step C: project all points into positive half-plane
    for i, m in enumerate(movables):
        t = m["target"]

        for obs in fixed_obstacles:
            c = obs["center"]
            # Vector from fixed obstacle center to movable target
            vec_to_target = t - c
            vec_len = np.linalg.norm(vec_to_target)

            # Skip if target is at the center (degenerate case)
            if vec_len < 1e-10:
                vec_len = 1e-10

            normal = vec_to_target / vec_len
            pts[i] = project_to_positive_side(pts[i], c, normal, min_dist=0.05)

    # Step D: resolve initial overlaps
    pts = resolve_initial_collisions(pts, movables, fixed_obstacles)

    return pts.reshape(-1)


def optimize(
    movables,
    fixed_obstacles,
    num_restarts=20,
    maxiter=400,
    placement_bounds=None,
    min_separation=0.2,
    margin_ratio=0.1,
    target_weight=100.0,
    normal_weight=100.0,
):
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

    # Create bounds for optimizer: list of (min, max) tuples for each variable
    n = len(movables)
    bounds = []
    for _ in range(n):
        bounds.append(placement_bounds[0])  # x bounds: (xmin, xmax)
        bounds.append(placement_bounds[1])  # y bounds: (ymin, ymax)

    # Create non-overlap constraints with adaptive margins
    overlap_constraints = create_non_overlap_constraints(
        movables, fixed_obstacles, min_separation, margin_ratio
    )

    # # Create positive side constraints (movables must be in front of fixed obstacles)
    # positive_side_constraints = create_positive_side_constraints(
    #     movables, fixed_obstacles
    # )

    # Combine all constraints
    # constraints = positive_side_constraints + overlap_constraints
    constraints = overlap_constraints

    # Helper function to check constraint feasibility
    def check_feasibility(x, constraints_list, verbose=False):
        """Check if a point satisfies all constraints."""
        violations = []
        for idx, constraint in enumerate(constraints_list):
            val = constraint["fun"](x)
            violations.append(val)
            if verbose and val < -1e-6:
                constraint_type = (
                    "overlap" if idx < len(overlap_constraints) else "positive-side"
                )
        min_val = min(violations) if violations else 0.0
        return min_val >= -1e-6, min_val, violations

    # Check if starting from targets is feasible
    x_targets = np.array([m["target"] for m in movables]).reshape(-1)
    target_feasible, target_min_const, _ = check_feasibility(
        x_targets, constraints, verbose=False
    )
    print(
        f"Target positions feasibility: {'FEASIBLE' if target_feasible else 'INFEASIBLE'} (min constraint: {target_min_const:.6f})"
    )

    # prev_solution = None  # Track feasible partial solutions across restarts

    batch_size = 1
    num_batches = (len(movables) + batch_size - 1) // batch_size  # Ceiling division
    x0 = np.array([m["target"] for m in movables]).reshape(-1)
    result = x0.copy()
    print(
        f"Processing {len(movables)} movables in {num_batches} batches of size {batch_size}"
    )

    # Create objective function for Stage 1 (with overlap penalty)
    def cost_func_stage1(x, batch_idx, batch_size):
        movables_subset = movables[
            batch_idx * batch_size : (batch_idx + 1) * batch_size
        ]

        fixed_obstacles_subset = []
        for movable in movables_subset:
            hostElementID = movable["HostElementId"]
            for obs in fixed_obstacles:
                if obs["ElementId"] == hostElementID:
                    fixed_obstacles_subset.append(obs)
                    break
                else:
                    fixed_obstacles_subset.append({"center": None})

        # Pre-compute polygon sizes for adaptive margins
        movable_sizes = [
            polygon_characteristic_size(m["verts"]) for m in movables_subset
        ]
        fixed_sizes = [
            polygon_characteristic_size(obs["verts"])
            if obs.get("center") is not None and "verts" in obs
            else 0.0
            for obs in fixed_obstacles
        ]

        # Calculate objective (target + normal alignment)
        obj_val = objective(
            x,
            movables_subset,
            fixed_obstacles_subset,
            fixed_obstacles,
            target_weight,
            normal_weight,
            overlap_weight=1000.0,
            min_separation=min_separation,
            margin_ratio=margin_ratio,
            movable_sizes=movable_sizes,
            fixed_sizes=fixed_sizes,
        )

        # Calculate overlap penalty (ACTIVE in Stage 1)
        overlap_val = overlap_penalty(
            x,
            movables_subset,
            fixed_obstacles,
            overlap_weight=1000.0,
            min_separation=min_separation,
            margin_ratio=margin_ratio,
            movable_sizes=movable_sizes,
            fixed_sizes=fixed_sizes,
        )

        return obj_val + overlap_val

    # Create objective function for Stage 2 (without overlap penalty)
    def cost_func_stage2(x, batch_idx, batch_size):
        movables_subset = movables[
            batch_idx * batch_size : (batch_idx + 1) * batch_size
        ]

        fixed_obstacles_subset = []
        for movable in movables_subset:
            hostElementID = movable["HostElementId"]
            for obs in fixed_obstacles:
                if obs["ElementId"] == hostElementID:
                    fixed_obstacles_subset.append(obs)
                    break
                else:
                    fixed_obstacles_subset.append({"center": None})

        # Pre-compute polygon sizes
        movable_sizes = [
            polygon_characteristic_size(m["verts"]) for m in movables_subset
        ]
        fixed_sizes = [
            polygon_characteristic_size(obs["verts"])
            if obs.get("center") is not None and "verts" in obs
            else 0.0
            for obs in fixed_obstacles
        ]

        # Only objective (target + normal alignment), NO overlap penalty
        obj_val = objective(
            x,
            movables_subset,
            fixed_obstacles_subset,
            fixed_obstacles,
            1.0,
            1.0,
            overlap_weight=0.0,  # Turn OFF overlap penalty
            min_separation=min_separation,
            margin_ratio=margin_ratio,
            movable_sizes=movable_sizes,
            fixed_sizes=fixed_sizes,
        )

        return obj_val

    for restart_idx in range(num_restarts):
        try:
            # Loop over each batch
            for batch_num in range(num_batches):
                batch_start = batch_num * batch_size
                batch_end = min(batch_start + batch_size, len(movables))
                actual_batch_size = batch_end - batch_start

                # ==================== STAGE 1: Soft Penalty Optimization ====================
                print(
                    "STAGE 1: Optimization with soft overlap penalties (no hard constraints)"
                )
                # Extract initial positions for this batch (2 coords per movable: x, y)
                x0_batch = x0[batch_start * 2 : batch_end * 2]

                # # Stage 1: NO constraints, only soft penalties
                # res_stage1 = minimize(
                #     cost_func_stage1,
                #     x0_batch,
                #     args=(batch_num, actual_batch_size),
                #     method="trust-constr",
                #     jac="3-point",
                #     # hess=BFGS(),
                #     bounds=bounds[batch_start * 2 : batch_end * 2],
                #     options=dict(
                #         verbose=3,
                #         xtol=1e-3,
                #         gtol=1e-3,
                #         barrier_tol=1e-3,
                #         initial_tr_radius=10.0,
                #         maxiter=20,
                #         finite_diff_rel_step=1e-3,
                #     ),
                # )

                # # Store Stage 1 result
                # result[batch_start * 2 : batch_end * 2] = res_stage1.x
                # print(f"Stage 1 - Batch {batch_num + 1} Result: {res_stage1.fun:.6e}")

                # ==================== STAGE 1.5: Geometry-based Projection ====================
                print(
                    f"Stage 1.5 - Projecting Batch {batch_num + 1} to resolve overlaps..."
                )

                # Create temporary fixed obstacles from previous batches to prevent overlaps with them
                temp_fixed = list(fixed_obstacles)
                for prev_idx in range(batch_start):
                    temp_fixed.append(
                        {
                            "verts": movables[prev_idx]["verts"],
                            "center": result[prev_idx * 2 : prev_idx * 2 + 2],
                            "RotationAngle": movables[prev_idx].get(
                                "RotationAngle", 0.0
                            ),
                        }
                    )

                # Project current batch positions
                x_batch = result[batch_start * 2 : batch_end * 2]
                x_projected = project_to_nonoverlap(
                    x_batch,
                    movables[batch_start:batch_end],
                    temp_fixed,
                    max_proj_iters=500,
                    min_separation=min_separation,
                )

                # Update result with projected positions
                result[batch_start * 2 : batch_end * 2] = x_projected
                print(f"Stage 1.5 - Batch {batch_num + 1} projection complete.")

        except Exception as e:
            print(f"Stage 1 - Restart {restart_idx + 1}: Optimization failed: {e}")
            continue

    # # ==================== STAGE 2: Hard Constraint Refinement ====================
    # print("\n" + "=" * 80)
    # print("STAGE 2: Refinement with hard overlap constraints (no penalty)")
    # print("=" * 80)

    # batch_size = len(movables)
    # num_batches = (len(movables) + batch_size - 1) // batch_size  # Ceiling division
    # # Use Stage 1 result as starting point for Stage 2
    # x0_stage2 = result.copy()

    # for restart_idx in range(1):  # Usually 1 restart is enough for Stage 2
    #     try:
    #         # Loop over each batch
    #         for batch_num in range(num_batches):
    #             batch_start = batch_num * batch_size
    #             batch_end = min(batch_start + batch_size, len(movables))
    #             actual_batch_size = batch_end - batch_start

    #             print(
    #                 f"\nStage 2 - Batch {batch_num + 1}/{num_batches} (movables {batch_start}-{batch_end - 1})"
    #             )

    #             # Extract Stage 1 solution as initial point for Stage 2
    #             x0_batch_stage2 = x0_stage2[batch_start * 2 : batch_end * 2]

    #             # Stage 2: WITH hard overlap constraints
    #             res_stage2 = minimize(
    #                 cost_func_stage2,
    #                 x0_batch_stage2,
    #                 args=(batch_num, actual_batch_size),
    #                 method="trust-constr",
    #                 jac="2-point",
    #                 hess=BFGS(),
    #                 bounds=bounds[batch_start * 2 : batch_end * 2],
    #                 constraints=constraints,  # Activate hard constraints
    #                 options=dict(
    #                     verbose=3,
    #                     xtol=1e-3,
    #                     gtol=1e-3,
    #                     barrier_tol=1e-3,
    #                     initial_tr_radius=10.0,
    #                     maxiter=20,
    #                     finite_diff_rel_step=1e-3,
    #                 ),
    #             )

    #             # Store Stage 2 final result
    #             result[batch_start * 2 : batch_end * 2] = res_stage2.x
    #             print(f"Stage 2 - Batch {batch_num + 1} Result: {res_stage2.fun:.6e}")

    #     except Exception as e:
    #         print(f"Stage 2 - Batch {batch_num + 1}: Optimization failed: {e}")
    #         print("Using Stage 1 result for this batch")
    #         continue

    # if result is None:
    #     print(
    #         "Warning: No feasible solution found. Try increasing num_restarts or adjusting min_separation."
    #     )

    # Calculate search radius for visualization (same as in create_non_overlap_constraints)
    movable_sizes = [polygon_characteristic_size(m["verts"]) for m in movables]
    valid_fixed_obstacles = [
        obs for obs in fixed_obstacles if obs.get("center") is not None
    ]
    fixed_sizes = [
        polygon_characteristic_size(obs["verts"]) for obs in valid_fixed_obstacles
    ]

    if len(movable_sizes) > 0 and len(fixed_sizes) > 0:
        max_movable_size = np.max(movable_sizes)
        max_fixed_size = np.max(fixed_sizes)
        search_radius = (max_movable_size) * 2
    else:
        search_radius = 0.0

    return result, x0, search_radius
