"""
Polygon utility functions for convex hull computation, collision detection, and geometry operations.
"""

import scipy.ndimage._ni_support
import numpy as np
from scipy.spatial import ConvexHull


def unpack_xy(xvec):
    """Unpack position vector into 2D points."""
    return xvec.reshape(-1, 2)


def sort_vertices_ccw(verts):
    """Sort vertices in counter-clockwise order around centroid."""
    # Compute centroid
    c = np.mean(verts, axis=0)
    # Compute angle of each vertex around centroid
    angles = np.arctan2(verts[:, 1] - c[1], verts[:, 0] - c[0])
    return verts[np.argsort(angles)]


def translate_polygon(verts, pos, angle=0.0):
    """
    Translate and rotate polygon vertices.

    Args:
        verts: Array of vertices (N, 2)
        pos: Position vector (2,)
        angle: Rotation angle in radians (counter-clockwise)

    Returns:
        Transformed vertices
    """
    if angle != 0.0:
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        # Rotate vertices
        verts_rotated = verts @ R.T
        return verts_rotated + pos
    return verts + pos


def polygon_characteristic_size(verts):
    """
    Calculate characteristic size of a polygon (useful for adaptive margins).
    Uses the average of width and height of bounding box.

    Args:
        verts: Array of polygon vertices

    Returns:
        Characteristic size (average of bounding box dimensions)
    """
    if len(verts) == 0:
        return 0.0

    # Get convex hull for consistent size calculation
    hull_verts = get_convex_hull_vertices(verts, closed=False)
    if len(hull_verts) == 0:
        return 0.0

    # Calculate bounding box
    min_x, max_x = np.min(hull_verts[:, 0]), np.max(hull_verts[:, 0])
    min_y, max_y = np.min(hull_verts[:, 1]), np.max(hull_verts[:, 1])

    width = max_x - min_x
    height = max_y - min_y

    # Return average dimension as characteristic size
    return (width + height) / 2.0


def get_convex_hull_vertices(verts, closed=False):
    """
    Compute convex hull of vertices and return ordered vertices.
    Handles any number of vertices and ensures proper polygon shape.

    Args:
        verts: Array of vertices
        closed: If True, append first vertex to close the polygon (for drawing)

    Returns:
        Array of vertices in counter-clockwise order (optionally closed)
    """
    if len(verts) < 3:
        # Need at least 3 points for a polygon
        if closed and len(verts) > 0:
            return np.vstack([verts, verts[0:1]])
        return verts

    try:
        # Compute convex hull
        hull = ConvexHull(verts)
        # Get vertices in counter-clockwise order
        hull_verts = verts[hull.vertices]
        if closed:
            # Close the polygon by appending first vertex
            return np.vstack([hull_verts, hull_verts[0:1]])
        return hull_verts
    except:
        # Fallback: if convex hull fails, sort vertices
        sorted_verts = sort_vertices_ccw(verts)
        if closed:
            return np.vstack([sorted_verts, sorted_verts[0:1]])
        return sorted_verts


def polygon_edges(verts):
    """Compute polygon edges from vertices using convex hull."""
    # Use convex hull vertices for edge calculation
    hull_verts = get_convex_hull_vertices(verts, closed=False)
    if len(hull_verts) < 2:
        return np.array([]).reshape(0, 2)
    return np.roll(hull_verts, -1, axis=0) - hull_verts


def normals_from_edges(edges):
    """Compute perpendicular normals from edges (rotated 90 degrees)."""
    n = np.zeros_like(edges)
    n[:,0] = -edges[:,1]
    n[:,1] = edges[:,0]
    # normalize
    lens = np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    return n / lens


def project_polygon(axis, verts):
    """Project polygon onto an axis and return min/max values."""
    proj = verts.dot(axis)
    return np.min(proj), np.max(proj)


def get_precomputed_geometry(verts, angle=0.0):
    """
    Precompute convex hull, rotated vertices, and normals for a polygon.
    This avoids redundant calculations during optimization.
    """
    hull = get_convex_hull_vertices(verts, closed=False)
    # Rotate hull once to world orientation (if angle is fixed)
    if angle != 0.0:
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        hull_rotated = hull @ R.T
    else:
        hull_rotated = hull.copy()

    edges = polygon_edges(hull_rotated)
    normals = normals_from_edges(edges)

    return {"hull": hull_rotated, "normals": normals}


def separating_distance_SAT_precomputed(
    hullA, hullB, normalsA, normalsB, return_normal=False
):
    """
    Fast, vectorized SAT implementation using precomputed hulls and normals.
    """
    # Combine normals from both polygons as axes
    axes = np.vstack([normalsA, normalsB])

    # Project both hulls onto all axes at once (vectorized)
    # hullA: (Na, 2), axes: (Naxes, 2) -> projA: (Na, Naxes)
    projA = hullA @ axes.T
    projB = hullB @ axes.T

    minA, maxA = projA.min(axis=0), projA.max(axis=0)
    minB, maxB = projB.min(axis=0), projB.max(axis=0)

    # Check for separation on each axis
    # separation > 0 means separated, < 0 means overlap depth
    seps = np.maximum(minB - maxA, minA - maxB)

    # If any separation is positive, the polygons are separated
    # The true separation distance is the maximum of these
    max_sep = np.max(seps)

    if max_sep >= 0:
        if not return_normal:
            return max_sep

        # For separation, we want the axis with the smallest positive separation
        # (the "tightest" separation axis)
        sep_indices = np.where(seps >= 0)[0]
        best_idx = sep_indices[np.argmin(seps[sep_indices])]

        # Determine normal direction: push A away from B
        axis = axes[best_idx]
        if minB[best_idx] - maxA[best_idx] > minA[best_idx] - maxB[best_idx]:
            # B is "ahead" of A along axis, push A back
            normal = -axis
        else:
            # A is "ahead" of B along axis, push A forward
            normal = axis

        return max_sep, normal, 0.0
    else:
        # All axes show overlap, polygons overlap
        # Penetration depth is the minimum overlap (smallest absolute value)
        # which is the maximum of the negative separations
        penetration = -max_sep

        if not return_normal:
            return max_sep  # returns negative value for overlap

        # Find axis with minimum penetration
        best_idx = np.argmax(seps)
        axis = axes[best_idx]

        # Determine direction: push A away from B
        if maxA[best_idx] - minB[best_idx] < maxB[best_idx] - minA[best_idx]:
            normal = -axis
        else:
            normal = axis

        return max_sep, normal, penetration


def separating_distance(polyA, polyB):
    return separating_distance_GJK_EPA(polyA, polyB)


def separating_distance_SAT(polyA, polyB, return_normal=False):
    """
    Compute separating distance along all SAT axes.
    If positive => separated; if negative => overlap depth.

    Uses Separating Axis Theorem (SAT) for collision detection.
    Uses convex hull vertices for consistent collision detection.

    Args:
        polyA: First polygon vertices
        polyB: Second polygon vertices
        return_normal: If True, also return the normal vector and penetration

    Returns:
        If return_normal=False: separation distance (float)
        If return_normal=True: tuple of (separation, normal, penetration)
    """
    # Get convex hull vertices for both polygons (consistent with polygon_edges)
    hullA = get_convex_hull_vertices(polyA, closed=False)
    hullB = get_convex_hull_vertices(polyB, closed=False)

    # Compute edges from convex hulls
    edgesA = polygon_edges(hullA)
    edgesB = polygon_edges(hullB)

    # Get normals from edges
    axes = []
    if len(edgesA) > 0:
        axes.extend(normals_from_edges(edgesA))
    if len(edgesB) > 0:
        axes.extend(normals_from_edges(edgesB))

    # If no edges, return a large separation
    if len(axes) == 0:
        if return_normal:
            return np.inf, np.array([1.0, 0.0]), 0.0
        return np.inf

    # Track minimum overlap depth (for penetration) and minimum separation (for distance)
    min_overlap = np.inf
    min_separation = np.inf
    best_overlap_axis = None
    best_overlap_sign = 1.0
    best_sep_axis = None
    best_sep_sign = 1.0

    for a in axes:
        a = a / (np.linalg.norm(a) + 1e-12)
        # Project using convex hull vertices for consistency
        minA, maxA = project_polygon(a, hullA)
        minB, maxB = project_polygon(a, hullB)

        # Check if intervals overlap on this axis
        if maxA > minB and maxB > minA:
            # Overlapping: compute overlap depth
            overlap_depth = min(maxA - minB, maxB - minA)
            if overlap_depth < min_overlap:
                min_overlap = overlap_depth
                best_overlap_axis = a.copy()
                # Determine direction: push A away from B
                # If maxA is closer to minB, push A in negative axis direction
                if maxA - minB < maxB - minA:
                    best_overlap_sign = -1.0
                else:
                    best_overlap_sign = 1.0
        else:
            # Separated on this axis: compute separation distance
            separation = max(minB - maxA, minA - maxB)
            if separation < min_separation:
                min_separation = separation
                best_sep_axis = a.copy()
                # Determine direction: push A away from B
                if minB - maxA > minA - maxB:
                    # B is to the right of A, push A to the left
                    best_sep_sign = -1.0
                else:
                    # A is to the right of B, push A to the right
                    best_sep_sign = 1.0

    # If we found any separation axis, polygons are separated
    if min_separation < np.inf:
        if return_normal:
            normal = best_sep_axis * best_sep_sign
            return min_separation, normal, 0.0
        return min_separation  # Positive: separation distance
    else:
        # All axes show overlap, polygons overlap
        if return_normal:
            normal = best_overlap_axis * best_overlap_sign
            return -min_overlap, normal, min_overlap
        return -min_overlap  # Negative: overlap depth


def separating_distance_GJK_EPA(polyA, polyB):
    """
    Signed distance between two convex polygons using GJK + EPA.
    Positive = separation distance.
    Negative = penetration depth.
    """

    def support(points, direction):
        idx = np.argmax(points @ direction)
        return points[idx]

    def support_minkowski(A, B, d):
        return support(A, d) - support(B, -d)

    def handle_line(simplex):
        A = simplex[-1]
        B = simplex[-2]
        AB = B - A
        AO = -A

        if np.dot(AB, AO) > 0:
            direction = np.array([AB[1], -AB[0]])  # perpendicular
            if np.dot(direction, AO) < 0:
                direction = -direction
            return False, direction
        else:
            simplex[:] = [A]
            return False, AO

    def handle_triangle(simplex):
        A, B, C = simplex[-1], simplex[-2], simplex[-3]
        AO = -A
        AB = B - A
        AC = C - A

        perpAB = np.array([AB[1], -AB[0]])
        if np.dot(perpAB, AC) > 0:
            perpAB = -perpAB
        if np.dot(perpAB, AO) > 0:
            simplex[:] = [A, B]
            return False, perpAB

        perpAC = np.array([AC[1], -AC[0]])
        if np.dot(perpAC, AB) > 0:
            perpAC = -perpAC
        if np.dot(perpAC, AO) > 0:
            simplex[:] = [A, C]
            return False, perpAC

        return True, None

    def update_simplex(simplex):
        if len(simplex) == 2:
            return handle_line(simplex)
        else:
            return handle_triangle(simplex)

    def gjk(A, B, max_iter=30):
        direction = np.array([1.0, 0.0])
        simplex = [support_minkowski(A, B, direction)]
        direction = -simplex[0]

        for _ in range(max_iter):
            new_point = support_minkowski(A, B, direction)
            if new_point @ direction <= 0:
                return False, simplex  # No collision → distance mode

            simplex.append(new_point)

            hit, direction = update_simplex(simplex)
            if hit:
                return True, simplex  # Collision detected

        return False, simplex

    def epa(A, B, simplex, max_iter=40):
        polytope = list(simplex)

        def edge_normal(a, b):
            e = b - a
            normal = np.array([e[1], -e[0]])
            normal = normal / (np.linalg.norm(normal) + 1e-12)
            if np.dot(normal, a) < 0:
                normal = -normal
            dist = np.dot(normal, a)
            return dist, normal

        for _ in range(max_iter):
            # Find closest edge to origin
            best_dist = np.inf
            best_normal = None
            best_i = None

            for i in range(len(polytope)):
                a = polytope[i]
                b = polytope[(i + 1) % len(polytope)]
                dist, normal = edge_normal(a, b)
                if dist < best_dist:
                    best_dist = dist
                    best_normal = normal
                    best_i = i

            # Get farthest point along normal
            p = support_minkowski(A, B, best_normal)
            d = np.dot(best_normal, p)

            # Convergence
            if d - best_dist < 1e-9:
                return best_dist

            # Insert point and continue
            polytope.insert(best_i + 1, p)

        return best_dist  # fallback

    def gjk_distance(A, B, simplex, max_iter=30):
        direction = -simplex[0]

        for _ in range(max_iter):
            p = support_minkowski(A, B, direction)
            if np.dot(p - simplex[0], direction) < 1e-9:
                return np.linalg.norm(direction)

            simplex.append(p)
            hit, direction = update_simplex(simplex)

            if hit:
                return 0.0

        return 0.0

    # Run GJK: returns (hit, closest simplex)
    hit, simplex = gjk(polyA, polyB)

    if not hit:
        # No collision ⇒ return Euclidean distance between shapes
        return gjk_distance(polyA, polyB, simplex)

    # If collision ⇒ compute penetration depth using EPA
    penetration_depth = epa(polyA, polyB, simplex)
    return -penetration_depth
