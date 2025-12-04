"""
Optimizer and cost functions for object placement optimization.
"""

import numpy as np
from scipy.optimize import minimize
from polygon_utils import (
    unpack_xy,
    translate_polygon,
    get_convex_hull_vertices,
    separating_distance,
    polygon_characteristic_size,
)


def objective(xvec, movables, fixed_obstacles, target_weight=100.0, normal_weight=10.0):
    """
    Calculate objective with two components:
    1. Distance from movables to their targets
    2. Distance from movables to normal lines from fixed obstacles

    The normal is computed from the line connecting fixed_obstacle center to movable target.

    Args:
        xvec: Flattened array of positions for all movable objects
        movables: List of movable object dictionaries with 'verts' and 'target' keys
        fixed_obstacles: List of fixed obstacle dictionaries with 'center' key
        target_weight: Weight for target distance term
        normal_weight: Weight for normal alignment term

    Returns:
        Total weighted objective value
    """
    pts = unpack_xy(xvec)
    val = 0.0

    # Component 1: Distance from movables to targets
    for i, p in enumerate(pts):
        t = movables[i]["target"]
        val += target_weight * np.sum((p - t) ** 2)

    # Component 2: Distance from movables to normal lines from fixed obstacles
    for i, p in enumerate(pts):
        t = movables[i]["target"]

        for obs in fixed_obstacles:
            center = obs["center"]

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


def create_non_overlap_constraints(
    movables, fixed_obstacles, min_separation=0.0, margin_ratio=0.1
):
    """
    Create constraint functions for non-overlap conditions with adaptive margins.

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

    # Constraint: movable-movable non-overlap
    n_movables = len(movables)
    for i in range(n_movables):
        for j in range(i + 1, n_movables):
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
                        movables[i_val]["verts"], pts[i_val]
                    )
                    B_translated = translate_polygon(
                        movables[j_val]["verts"], pts[j_val]
                    )
                    sep = separating_distance(A_translated, B_translated)
                    # Constraint satisfied when sep >= margin
                    # Return value must be >= 0 for constraint to be satisfied
                    return sep - margin

                return constraint

            constraints.append(
                {"type": "ineq", "fun": make_constraint_movable(i, j, adaptive_margin)}
            )

    # Constraint: movable-fixed non-overlap
    for i in range(n_movables):
        for obs_idx, obs in enumerate(fixed_obstacles):
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
                        movables[i_val]["verts"], pts[i_val]
                    )
                    B_translated = translate_polygon(
                        obs_val["verts"], obs_val["center"]
                    )
                    sep = separating_distance(A_translated, B_translated)
                    # Constraint satisfied when sep >= margin
                    return sep - margin

                return constraint

            constraints.append(
                {"type": "ineq", "fun": make_constraint_fixed(i, obs, adaptive_margin)}
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
                    # print("dot product", np.dot(vec_to_movable_norm, normal_vec))
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
                    A = translate_polygon(movables[i]["verts"], pts[i])
                    B = translate_polygon(movables[j]["verts"], pts[j])
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
                    A = translate_polygon(movables[i]["verts"], pts[i])
                    B = translate_polygon(obs["verts"], obs["center"])
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
    min_separation=0.0,
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

    # Set default bounds if not provided
    if placement_bounds is None:
        placement_bounds = ((-10.0, 10.0), (-10.0, 10.0))

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

    # Create positive side constraints (movables must be in front of fixed obstacles)
    positive_side_constraints = create_positive_side_constraints(
        movables, fixed_obstacles
    )

    # Combine all constraints
    constraints = positive_side_constraints + overlap_constraints

    print(f"Created {len(overlap_constraints)} non-overlap constraints")
    print(f"Created {len(positive_side_constraints)} positive-side constraints")
    print(f"Total constraints: {len(constraints)}")

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
                print(f"  Constraint {idx} ({constraint_type}) violated: {val:.6f}")
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

    # Create objective function
    def cost_func(x):
        return objective(x, movables, fixed_obstacles, target_weight, normal_weight)

    prev_solution = None  # Track feasible partial solutions across restarts

    for restart_idx in range(num_restarts):
        # Better initialization: try to start near targets but check feasibility
        max_init_attempts = 10
        x0 = None

        for init_attempt in range(max_init_attempts):
            x0_candidate = generate_initial_point(movables, fixed_obstacles, init_attempt, prev_solution=None)


            # Check if initial point is feasible
            is_feasible, min_const, violations = check_feasibility(
                x0_candidate, constraints, verbose=False
            )
            if is_feasible or init_attempt == max_init_attempts - 1:
                x0 = x0_candidate
                if not is_feasible:
                    print(
                        f"Restart {restart_idx + 1}: Starting from infeasible point (min constraint: {min_const:.6f})"
                    )
                break

        if x0 is None:
            x0 = np.array([m["target"] for m in movables]).reshape(-1)

        try:
            res = minimize(
                cost_func,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                # options={"maxiter": maxiter, "disp": True, "ftol": 1e-8},
            )

            # Check if solution is feasible (all constraints satisfied)
            is_feasible, min_constraint, violations = check_feasibility(
                res.x, constraints, verbose=False
            )

            # Only accept feasible solutions
            if is_feasible:
                if best is None or res.fun < best.fun:
                    best = res
                    best_initial = x0.copy()
                    print(
                        f"Restart {restart_idx + 1}: Found feasible solution with cost {res.fun:.6f}, "
                        f"min constraint: {min_constraint:.6f}"
                    )
                    prev_solution = res.x.copy()
                    # If we found a very good solution, break early
                    if res.fun < 1e-3:
                        break
            else:
                # Debug: show which constraints are most violated
                print(
                    f"Restart {restart_idx + 1}: Infeasible solution (min constraint: {min_constraint:.6f})"
                )
                print(f"  Most violated constraints:")
                sorted_violations = sorted(
                    enumerate(violations), key=lambda x: x[1]
                )[:5]
                for idx, val in sorted_violations:
                    constraint_type = (
                        "overlap"
                        if idx < len(overlap_constraints)
                        else "positive-side"
                    )
                    print(f"    Constraint {idx} ({constraint_type}): {val:.6f}")


        except Exception as e:
            print(f"Restart {restart_idx + 1}: Optimization failed: {e}")
            continue

    if best is None:
        print(
            "Warning: No feasible solution found. Try increasing num_restarts or adjusting min_separation."
        )

    return best, best_initial
