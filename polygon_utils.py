"""
Polygon utility functions for convex hull computation, collision detection, and geometry operations.
"""

import numpy as np
# from scipy.spatial import ConvexHull


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

    # try:
    #     # Compute convex hull
    #     hull = ConvexHull(verts)
    #     # Get vertices in counter-clockwise order
    #     hull_verts = verts[hull.vertices]
    #     if closed:
    #         # Close the polygon by appending first vertex
    #         return np.vstack([hull_verts, hull_verts[0:1]])
    #     return hull_verts
    # except:
    #     # Fallback: if convex hull fails, sort vertices
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
    Fast, vectorized cv implementation using precomputed hulls and normals.
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
