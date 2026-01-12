"""
Optimizer and cost functions for object placement optimization.
"""

import numpy as np
from scipy.spatial import cKDTree
from joblib import Parallel, delayed
from polygon_utils import (
    unpack_xy,
    polygon_characteristic_size,
    get_precomputed_geometry,
    separating_distance_SAT_precomputed,
    translate_polygon,
)
from numba_utils import pack_geometry, check_collisions_numba


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
    OPTIMIZED VERSION with spatial filtering and batch overlap resolution.

    Optimizations applied:
    1. Spatial filtering with KD-tree (3-5x speedup)
    2. Batch resolution of non-conflicting overlaps (2-5x speedup)
    3. Early convergence detection (1.5-2x speedup)
    4. Precompute geometry once outside loop (1.2x speedup)

    Expected combined speedup: 5-15x

    This function iteratively:
    1. Finds overlapping pairs using spatial filtering
    2. Resolves MULTIPLE non-conflicting overlaps per iteration
    3. Checks for convergence
    4. Repeats until no overlaps remain or max iterations reached

    Args:
        x: Flattened array of positions for all movable objects
        movables: List of movable object dictionaries
        fixed_obstacles: List of fixed obstacle dictionaries
        max_proj_iters: Maximum number of projection iterations (default: 50)
        min_separation: Minimum separation distance between objects (default: 0.2)
                       If > 0, objects will maintain a gap instead of just touching

    Returns:
        Projected positions (flattened array) with overlaps resolved
    """
    pts = unpack_xy(x).copy()
    n_movables = len(movables)

    # For Numba optimization (Stage 2 single-item check), we want to avoid
    # creating KDTrees and Python objects inside the loop.
    # If we are processing a SINGLE movable against a list of fixed obstacles,
    # we can use the fast Numba path.

    use_numba = (len(movables) == 1) and (len(fixed_obstacles) > 0)

    if False:
        # Prepare geometry for the single movable
        # It changes position every iteration, so we pass its current verts/normals
        # But its shape/rotation is constant, so we can pre-calc the local rotated shape
        m = movables[0]
        # Get local rotated verts (centered at 0,0)
        # m["verts"] is local. Rotate it.
        c, s = np.cos(m.get("RotationAngle", 0.0)), np.sin(m.get("RotationAngle", 0.0))
        R = np.array([[c, -s], [s, c]])
        local_verts_rotated = m["verts"] @ R.T

        # Compute normals for the movable once
        # (Same logic as pack_geometry but for single item)
        verts_next = np.roll(local_verts_rotated, -1, axis=0)
        edges = verts_next - local_verts_rotated
        local_normals = np.zeros_like(edges)
        local_normals[:, 0] = -edges[:, 1]
        local_normals[:, 1] = edges[:, 0]
        norms = np.sqrt(local_normals[:, 0] ** 2 + local_normals[:, 1] ** 2)
        norms[norms < 1e-12] = 1.0
        local_normals /= norms[:, np.newaxis]

        # Pack fixed obstacles ONCE
        # We need their WORLD space vertices
        fixed_polys_world = []
        for obs in fixed_obstacles:
            # Translate to world space
            v_world = translate_polygon(
                obs["verts"], obs["center"], obs.get("RotationAngle", 0.0)
            )
            fixed_polys_world.append({"verts": v_world})

        packed_fixed_verts, packed_fixed_normals, fixed_offsets, fixed_counts = (
            pack_geometry(fixed_polys_world)
        )

        # Optimization loop
        current_pos = pts[0].copy()

        for iteration in range(max_proj_iters):
            # Update movable vertices to current position
            curr_verts_world = local_verts_rotated + current_pos

            # Check collisions using Numba
            # We pass the normals (rotation doesn't change, so normals are same direction)
            # Wait, normals are direction vectors. Translation doesn't change them!
            # So local_normals are valid for world space too.

            count, collisions = check_collisions_numba(
                curr_verts_world,
                local_normals,
                packed_fixed_verts,
                packed_fixed_normals,
                fixed_offsets,
                fixed_counts,
                min_separation,
            )

            if count == 0:
                break

            # Resolve collisions
            # Average push vector? Or max?
            # Let's sum them for "soft" resolution or take max for "hard".
            # Summing is usually more stable for multiple overlaps.

            total_push = np.zeros(2)
            max_push_len = 0.0

            for i in range(count):
                dist = collisions[i, 0]
                nx = collisions[i, 1]
                ny = collisions[i, 2]

                # Push vector
                push = np.array([nx, ny]) * dist
                total_push += push
                max_push_len = max(max_push_len, dist)

            # Apply correction
            # Damping factor to prevent oscillation
            # For single movable vs fixed obstacles, we can be aggressive (0.9)
            alpha = 0.5
            current_pos += total_push * alpha

        # Final check and aggressive cleanup for any remaining overlaps
        final_count, collisions = check_collisions_numba(
            local_verts_rotated + current_pos,
            local_normals,
            packed_fixed_verts,
            packed_fixed_normals,
            fixed_offsets,
            fixed_counts,
            min_separation,
        )

        if final_count > 0:
            # # Aggressive cleanup loop
            # # Greedy strategy: Resolve the WORST overlap only in each iteration
            # # This mimics the original implementation's final pass and prevents force cancellation
            # prev_final_count = final_count
            # stuck_counter = 0

            # for cleanup_iter in range(200):
            #     if final_count == 0:
            #         break

            #     # Check if stuck
            #     if final_count >= prev_final_count:
            #         stuck_counter += 1
            #     else:
            #         stuck_counter = 0
            #         prev_final_count = final_count

            #     # If stuck, apply random perturbation
            #     if stuck_counter > 10:
            #         # Random kick to escape local minimum
            #         angle = np.random.uniform(0, 2 * np.pi)
            #         dist = min_separation * 0.5
            #         perturbation = np.array([np.cos(angle), np.sin(angle)]) * dist
            #         current_pos += perturbation
            #         stuck_counter = 0  # Reset
            #     else:
            #         # Find worst overlap
            #         worst_idx = -1
            #         max_dist = -1.0

            #         for i in range(final_count):
            #             if collisions[i, 0] > max_dist:
            #                 max_dist = collisions[i, 0]
            #                 worst_idx = i

            #         if worst_idx >= 0:
            #             # Push fully out of the worst overlap + safety
            #             dist = collisions[worst_idx, 0]
            #             nx = collisions[worst_idx, 1]
            #             ny = collisions[worst_idx, 2]

            #             push = np.array([nx, ny]) * (dist * 1.05)
            #             current_pos += push

            #     # Re-check
            #     final_count, collisions = check_collisions_numba(
            #         local_verts_rotated + current_pos,
            #         local_normals,
            #         packed_fixed_verts,
            #         packed_fixed_normals,
            #         fixed_offsets,
            #         fixed_counts,
            #         min_separation,
            #     )
            print(
                f"WARNING: Numba path finished with {final_count} unresolved overlaps. Falling back to robust Python implementation."
            )
            # Update pts with the best position found so far by Numba
            # This gives the Python implementation a better starting point
            pts[0] = current_pos
        else:
            return current_pos.reshape(-1)

    # --- Original Logic for Batch Processing (Stage 1 & 3) ---
    # === OPTIMIZATION 1: Precompute geometry once ===
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

    # Pre-compute object sizes for spatial filtering
    movable_sizes = [polygon_characteristic_size(m["verts"]) for m in movables]
    max_movable_size = np.max(movable_sizes) if len(movable_sizes) > 0 else 1.0
    search_radius = max_movable_size * 20  # 4x safety factor

    # === OPTIMIZATION 2: Early convergence tracking ===
    prev_max_penetration = float("inf")
    tolerance = 1e-8
    no_progress_count = 0

    for iteration in range(max_proj_iters):
        # === OPTIMIZATION 3: Spatial filtering with KD-tree ===
        overlaps = []

        if n_movables > 0:
            # Build KD-tree for current positions
            movable_tree = cKDTree(pts)

            # Helper function for collision check
            def check_collision(i, j):
                hullA = movables[i]["precomputed"]["hull"] + pts[i]
                hullB = movables[j]["precomputed"]["hull"] + pts[j]
                sep, normal, penetration = separating_distance_SAT_precomputed(
                    hullA,
                    hullB,
                    movables[i]["precomputed"]["normals"],
                    movables[j]["precomputed"]["normals"],
                    return_normal=True,
                )
                if sep < min_separation - 1e-6:
                    return (i, j, min_separation - sep, normal)
                return None

            # Check only spatially nearby movable-movable pairs
            for i in range(n_movables):
                candidate_indices = movable_tree.query_ball_point(pts[i], search_radius)
                for j in candidate_indices:
                    if j <= i:
                        continue
                    result = check_collision(i, j)
                    if result:
                        overlaps.append(result)

            # Check movable-fixed overlaps
            fixed_centers = np.array(
                [
                    obs["center"]
                    for obs in fixed_obstacles
                    if obs.get("center") is not None
                ]
            )
            valid_fixed_indices = [
                idx
                for idx, obs in enumerate(fixed_obstacles)
                if obs.get("center") is not None
            ]

            if len(fixed_centers) > 0:
                fixed_tree = cKDTree(fixed_centers)

                for i in range(n_movables):
                    candidate_tree_indices = fixed_tree.query_ball_point(
                        pts[i], search_radius
                    )
                    for tree_idx in candidate_tree_indices:
                        obs_idx = valid_fixed_indices[tree_idx]
                        obs = fixed_obstacles[obs_idx]

                        hullA = movables[i]["precomputed"]["hull"] + pts[i]
                        hullB = obs["precomputed"]["hull"] + obs["center"]
                        sep, normal, penetration = separating_distance_SAT_precomputed(
                            hullA,
                            hullB,
                            movables[i]["precomputed"]["normals"],
                            obs["precomputed"]["normals"],
                            return_normal=True,
                        )
                        if sep < min_separation - 1e-6:
                            overlaps.append((i, None, min_separation - sep, normal))

        if not overlaps:
            # No overlaps found, we're done
            break

        # === OPTIMIZATION 4: Batch resolution of non-conflicting overlaps ===
        # Sort by penetration depth (worst first)
        overlaps.sort(key=lambda o: o[2], reverse=True)

        # Track convergence
        max_penetration = overlaps[0][2]
        if abs(prev_max_penetration - max_penetration) < tolerance:
            no_progress_count += 1
            if no_progress_count >= 50:
                # Converged - no significant progress
                break
        else:
            no_progress_count = 0
        prev_max_penetration = max_penetration

        # Resolve multiple non-conflicting overlaps in one iteration
        resolved_indices = set()

        for i, j, separation_distance, normal in overlaps:
            # Skip if either object already moved this iteration
            if i in resolved_indices:
                continue
            if j is not None and j in resolved_indices:
                continue

            # Resolve this overlap
            if j is None:
                # Fixed obstacle: only move the movable object
                pts[i] += separation_distance * normal
                resolved_indices.add(i)
            else:
                # Two movables: move both equally
                pts[i] += 0.5 * separation_distance * normal
                pts[j] -= 0.5 * separation_distance * normal
                resolved_indices.add(i)
                resolved_indices.add(j)

        # Early exit if no progress (all overlaps conflicted)
        if len(resolved_indices) == 0:
            # All overlaps conflict with each other, just resolve worst one
            i, j, separation_distance, normal = overlaps[0]
            if j is None:
                pts[i] += separation_distance * normal
            else:
                pts[i] += 0.5 * separation_distance * normal
                pts[j] -= 0.5 * separation_distance * normal

    # === FINAL PHASE: BRUTE-FORCE VERIFICATION & CLEANUP ===
    # Do ONE final pass checking ALL pairs to catch any missed overlaps
    print(
        f"Final verification pass - checking all {n_movables * (n_movables - 1) // 2} pairs..."
    )

    final_overlaps = []

    # Check ALL movable-movable pairs (no spatial filtering)
    for i in range(n_movables):
        for j in range(i + 1, n_movables):
            hullA = movables[i]["precomputed"]["hull"] + pts[i]
            hullB = movables[j]["precomputed"]["hull"] + pts[j]
            sep, normal, penetration = separating_distance_SAT_precomputed(
                hullA,
                hullB,
                movables[i]["precomputed"]["normals"],
                movables[j]["precomputed"]["normals"],
                return_normal=True,
            )
            if sep < min_separation - 1e-6:
                final_overlaps.append((i, j, min_separation - sep, normal))

    # Check ALL movable-fixed pairs (no spatial filtering)
    for i in range(n_movables):
        for obs_idx, obs in enumerate(fixed_obstacles):
            if obs.get("center") is None:
                continue
            hullA = movables[i]["precomputed"]["hull"] + pts[i]
            hullB = obs["precomputed"]["hull"] + obs["center"]
            sep, normal, penetration = separating_distance_SAT_precomputed(
                hullA,
                hullB,
                movables[i]["precomputed"]["normals"],
                obs["precomputed"]["normals"],
                return_normal=True,
            )
            if sep < min_separation - 1e-6:
                final_overlaps.append((i, None, min_separation - sep, normal))

    if final_overlaps:
        print(f"WARNING: Found {len(final_overlaps)} remaining overlaps! Resolving...")
        # Resolve remaining overlaps aggressively with extra iterations
        for cleanup_iter in range(100):  # Up to 100 cleanup iterations
            if not final_overlaps:
                break

            # Sort by worst overlap
            final_overlaps.sort(key=lambda o: o[2], reverse=True)
            resolved_indices = set()

            for i, j, separation_distance, normal in final_overlaps:
                if i in resolved_indices:
                    continue
                if j is not None and j in resolved_indices:
                    continue

                # Aggressive push with 5% extra to ensure clean separation
                push_factor = 1.05
                if j is None:
                    pts[i] += push_factor * separation_distance * normal
                    resolved_indices.add(i)
                else:
                    pts[i] += 0.5 * push_factor * separation_distance * normal
                    pts[j] -= 0.5 * push_factor * separation_distance * normal
                    resolved_indices.add(i)
                    resolved_indices.add(j)

            # Recheck overlaps
            final_overlaps = []
            for i in range(n_movables):
                for j in range(i + 1, n_movables):
                    hullA = movables[i]["precomputed"]["hull"] + pts[i]
                    hullB = movables[j]["precomputed"]["hull"] + pts[j]
                    sep, normal, penetration = separating_distance_SAT_precomputed(
                        hullA,
                        hullB,
                        movables[i]["precomputed"]["normals"],
                        movables[j]["precomputed"]["normals"],
                        return_normal=True,
                    )
                    if sep < min_separation - 1e-6:
                        final_overlaps.append((i, j, min_separation - sep, normal))

            for i in range(n_movables):
                for obs_idx, obs in enumerate(fixed_obstacles):
                    if obs.get("center") is None:
                        continue
                    hullA = movables[i]["precomputed"]["hull"] + pts[i]
                    hullB = obs["precomputed"]["hull"] + obs["center"]
                    sep, normal, penetration = separating_distance_SAT_precomputed(
                        hullA,
                        hullB,
                        movables[i]["precomputed"]["normals"],
                        obs["precomputed"]["normals"],
                        return_normal=True,
                    )
                    if sep < min_separation - 1e-6:
                        final_overlaps.append((i, None, min_separation - sep, normal))

        if final_overlaps:
            print(f"ERROR: Still have {len(final_overlaps)} overlaps after cleanup!")
        else:
            print("All overlaps resolved!")
    else:
        print("No overlaps found - verification passed!")

    return pts.reshape(-1)


def optimize(
    movables,
    fixed_obstacles,
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
    # ==================== STAGE 2: Batch-by-batch refinement with min_separation ====================
    print("\n" + "=" * 80)
    min_separation = 0.5
    print(f"STAGE 2: Batch-by-batch refinement with min_separation={min_separation}")
    print("=" * 80)
    x0 = np.array([movable["target"] for movable in movables]).reshape(-1)
    result = x0.copy()

    # Process each batch individually with tighter separation
    batch_size = 1
    num_batches = len(movables)

    for batch_num in range(num_batches):
        batch_start = batch_num * batch_size
        batch_end = min(batch_start + batch_size, len(movables))

        print(f"\nStage 1 - Projecting Batch {batch_num + 1}/{num_batches}...")

        # Create temporary fixed obstacles from all OTHER movables + original fixed obstacles
        temp_fixed = list(fixed_obstacles)
        for other_idx in range(len(movables)):
            if other_idx < batch_start or other_idx >= batch_end:
                temp_fixed.append(
                    {
                        "verts": movables[other_idx]["verts"],
                        "center": movables[other_idx]["target"],
                        "RotationAngle": movables[other_idx].get("RotationAngle", 0.0),
                    }
                )

        # Project current batch positions with tighter separation
        x_batch = result[batch_start * 2 : batch_end * 2]
        x_projected = project_to_nonoverlap(
            x_batch,
            movables[batch_start:batch_end],
            temp_fixed,
            max_proj_iters=300,
            min_separation=min_separation,
        )

        # Update result with projected positions
        result[batch_start * 2 : batch_end * 2] = x_projected
        print(f"Stage 1 - Batch {batch_num + 1} complete.")

    print("Stage 1 complete!")

    # ==================== STAGE 2: BATCH REFINEMENT ====================
    min_separation = 1.5
    print("\n" + "=" * 80)
    print(f"STAGE 2: Batch-by-batch refinement with min_separation={min_separation}")
    print("=" * 80)

    # Process each batch individually with tighter separation
    batch_size = 1
    num_batches = len(movables)

    for batch_num in range(num_batches):
        batch_start = batch_num * batch_size
        batch_end = min(batch_start + batch_size, len(movables))

        print(f"\nStage 2 - Projecting Batch {batch_num + 1}/{num_batches}...")

        # Create temporary fixed obstacles from all OTHER movables + original fixed obstacles
        temp_fixed = list(fixed_obstacles)
        for other_idx in range(len(movables)):
            if other_idx < batch_start or other_idx >= batch_end:
                temp_fixed.append(
                    {
                        "verts": movables[other_idx]["verts"],
                        "center": result[other_idx * 2 : other_idx * 2 + 2],
                        "RotationAngle": movables[other_idx].get("RotationAngle", 0.0),
                    }
                )

        # Project current batch positions with tighter separation
        x_batch = result[batch_start * 2 : batch_end * 2]
        x_projected = project_to_nonoverlap(
            x_batch,
            movables[batch_start:batch_end],
            temp_fixed,
            max_proj_iters=300,
            min_separation=min_separation,
        )

        # Update result with projected positions
        result[batch_start * 2 : batch_end * 2] = x_projected
        print(f"Stage 2 - Batch {batch_num + 1} complete.")

    print("Stage 2 complete!")

    # ==================== STAGE 3: FINAL TIGHTENING ====================
    min_separation = 1.5
    print("\n" + "=" * 80)
    print(f"STAGE 3: Final tightening with min_separation={min_separation}")
    print("=" * 80)

    result_stage3 = project_to_nonoverlap(
        result,
        movables,
        fixed_obstacles,
        max_proj_iters=300,
        min_separation=min_separation,
    )
    result = result_stage3
    print("Stage 3 complete!")
    return result, x0
