"""
Polygon utility functions for convex hull computation, collision detection, and geometry operations.
"""

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
        a = a / (np.linalg.norm(a)+1e-12)
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

