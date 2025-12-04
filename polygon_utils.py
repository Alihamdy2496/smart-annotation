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
    angles = np.arctan2(verts[:,1] - c[1], verts[:,0] - c[0])
    return verts[np.argsort(angles)]


def translate_polygon(verts, pos):
    """Translate polygon vertices by position vector."""
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


def separating_distance(polyA, polyB):
    return separating_distance_GJK_EPA(polyA, polyB)


def separating_distance_SAT(polyA, polyB):
    """
    Compute separating distance along all SAT axes.
    If positive => separated; if negative => overlap depth.

    Uses Separating Axis Theorem (SAT) for collision detection.
    Uses convex hull vertices for consistent collision detection.
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
        return np.inf

    # Track maximum overlap depth (negative) and minimum separation (positive)
    max_overlap = -np.inf
    min_separation = np.inf

    for a in axes:
        a = a / (np.linalg.norm(a) + 1e-12)
        # Project using convex hull vertices for consistency
        minA, maxA = project_polygon(a, hullA)
        minB, maxB = project_polygon(a, hullB)

        # Check if intervals overlap on this axis
        if maxA >= minB and maxB >= minA:
            # Overlapping: compute overlap depth
            overlap_depth = min(maxA - minB, maxB - minA)
            max_overlap = max(max_overlap, overlap_depth)
        else:
            # Separated on this axis: compute separation distance
            # When separated, one of these will be positive (the actual separation)
            separation = max(minB - maxA, minA - maxB)
            min_separation = min(min_separation, separation)

    # If we found any separation axis, polygons are separated
    if min_separation < np.inf:
        return min_separation  # Positive: separation distance
    else:
        # All axes show overlap, polygons overlap
        return -max_overlap  # Negative: overlap depth


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
