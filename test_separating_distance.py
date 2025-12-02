"""
Test script to visualize and understand the separating_distance function.
Creates plots showing polygons and their separation distances.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from polygon_utils import (
    translate_polygon, get_convex_hull_vertices, separating_distance
)


def plot_polygons_with_separation(polyA_verts, polyB_verts, posA, posB, title="", ax=None):
    """
    Plot two polygons and show their separation distance.
    
    Args:
        polyA_verts: Vertices of polygon A (local coordinates)
        polyB_verts: Vertices of polygon B (local coordinates)
        posA: Position of polygon A
        posB: Position of polygon B
        title: Title for the plot
        ax: Matplotlib axis (if None, creates new figure)
    """
    # Translate polygons
    A_translated = translate_polygon(polyA_verts, posA)
    B_translated = translate_polygon(polyB_verts, posB)
    
    # Compute separation distance
    sep = separating_distance(A_translated, B_translated)
    
    # Get convex hull vertices for visualization
    A_hull = get_convex_hull_vertices(A_translated, closed=True)
    B_hull = get_convex_hull_vertices(B_translated, closed=True)
    
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot polygon A
    polygon_A = Polygon(A_hull[:-1], closed=True, facecolor='blue', 
                       alpha=0.4, edgecolor='darkblue', linewidth=2)
    ax.add_patch(polygon_A)
    ax.plot(A_hull[:,0], A_hull[:,1], 'darkblue', linewidth=1.5)
    
    # Plot polygon B
    polygon_B = Polygon(B_hull[:-1], closed=True, facecolor='red', 
                       alpha=0.4, edgecolor='darkred', linewidth=2)
    ax.add_patch(polygon_B)
    ax.plot(B_hull[:,0], B_hull[:,1], 'darkred', linewidth=1.5)
    
    # Calculate centroids
    A_centroid = np.mean(A_hull[:-1], axis=0)
    B_centroid = np.mean(B_hull[:-1], axis=0)
    
    # Draw line between centroids
    ax.plot([A_centroid[0], B_centroid[0]], 
            [A_centroid[1], B_centroid[1]], 
            'g--', linewidth=1, alpha=0.5, label='Centroid line')
    
    # Add text with separation distance
    mid_point = (A_centroid + B_centroid) / 2
    color = 'red' if sep < 0 else 'green'
    ax.text(mid_point[0], mid_point[1], 
            f'sep = {sep:.4f}', 
            fontsize=12, fontweight='bold', color=color,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Mark positions
    ax.plot(posA[0], posA[1], 'bo', markersize=8, label='Pos A')
    ax.plot(posB[0], posB[1], 'ro', markersize=8, label='Pos B')
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'{title}\nSeparation: {sep:.6f} {"(OVERLAP)" if sep < 0 else "(SEPARATED)"}', 
                fontsize=12, fontweight='bold')
    ax.legend()
    
    return ax, sep


def test_scenarios():
    """Test various scenarios of polygon separation."""
    
    # Define test polygons
    square = np.array([[0.25, 0.25], [-0.25, 0.25], [-0.25, -0.25], [0.25, -0.25]])
    rectangle = np.array([[0.3, 0.2], [-0.3, 0.2], [-0.3, -0.2], [0.3, -0.2]])
    
    # Test scenarios
    scenarios = [
        {
            'name': '1. Fully Overlapping',
            'polyA': square,
            'polyB': square,
            'posA': np.array([0.0, 0.0]),
            'posB': np.array([0.0, 0.0]),
        },
        {
            'name': '2. Partially Overlapping',
            'polyA': square,
            'polyB': square,
            'posA': np.array([0.0, 0.0]),
            'posB': np.array([0.3, 0.3]),
        },
        {
            'name': '3. Just Touching',
            'polyA': square,
            'polyB': square,
            'posA': np.array([0.0, 0.0]),
            'posB': np.array([0.5, 0.0]),
        },
        {
            'name': '4. Separated',
            'polyA': square,
            'polyB': square,
            'posA': np.array([0.0, 0.0]),
            'posB': np.array([1.0, 1.0]),
        },
        {
            'name': '5. Different Shapes Overlapping',
            'polyA': square,
            'polyB': rectangle,
            'posA': np.array([0.0, 0.0]),
            'posB': np.array([0.2, 0.2]),
        },
        {
            'name': '6. Close but Separated',
            'polyA': square,
            'polyB': square,
            'posA': np.array([0.0, 0.0]),
            'posB': np.array([0.6, 0.0]),
        },
    ]
    
    # Create subplots
    n_scenarios = len(scenarios)
    cols = 3
    rows = (n_scenarios + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows))
    if n_scenarios == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    results = []
    
    for idx, scenario in enumerate(scenarios):
        ax = axes[idx]
        _, sep = plot_polygons_with_separation(
            scenario['polyA'], scenario['polyB'],
            scenario['posA'], scenario['posB'],
            scenario['name'], ax
        )
        results.append({
            'scenario': scenario['name'],
            'separation': sep,
            'overlapping': sep < 0
        })
    
    # Hide unused subplots
    for idx in range(n_scenarios, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('test_separating_distance.png', dpi=150, bbox_inches='tight')
    print("\n" + "="*80)
    print("SEPARATION DISTANCE TEST RESULTS")
    print("="*80)
    for result in results:
        status = "OVERLAPPING" if result['overlapping'] else "SEPARATED"
        print(f"{result['scenario']:30s} | sep = {result['separation']:10.6f} | {status}")
    print("="*80)
    print("\nNote: Negative values indicate overlap, positive values indicate separation.")
    print("The magnitude represents the overlap depth or separation distance.")
    plt.show()


def test_distance_sweep():
    """Test separation distance as one polygon moves relative to another."""
    
    square = np.array([[0.25, 0.25], [-0.25, 0.25], [-0.25, -0.25], [0.25, -0.25]])
    
    # Sweep polygon B along x-axis
    posA = np.array([0.0, 0.0])
    x_positions = np.linspace(-1.0, 1.0, 50)
    separations = []
    
    for x_pos in x_positions:
        posB = np.array([x_pos, 0.0])
        A_translated = translate_polygon(square, posA)
        B_translated = translate_polygon(square, posB)
        sep = separating_distance(A_translated, B_translated)
        separations.append(sep)
    
    # Plot separation vs position
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Separation distance curve
    ax1.plot(x_positions, separations, 'b-', linewidth=2, label='Separation Distance')
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=1, label='Zero (touching)')
    ax1.fill_between(x_positions, 0, separations, where=np.array(separations) < 0, 
                     alpha=0.3, color='red', label='Overlap Region')
    ax1.fill_between(x_positions, 0, separations, where=np.array(separations) > 0, 
                     alpha=0.3, color='green', label='Separation Region')
    ax1.set_xlabel('Position of Polygon B (x-coordinate)', fontsize=12)
    ax1.set_ylabel('Separation Distance', fontsize=12)
    ax1.set_title('Separation Distance vs Position (Polygon B moving along x-axis)', 
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Visualize key positions
    key_positions = [-0.8, -0.5, -0.25, 0.0, 0.25, 0.5, 0.8]
    n_keys = len(key_positions)
    cols = 4
    rows = (n_keys + cols - 1) // cols
    
    for idx, x_pos in enumerate(key_positions):
        row = idx // cols
        col = idx % cols
        if row < rows:
            # Create subplot
            if idx == 0:
                ax_sub = plt.subplot2grid((rows, cols), (row, col), fig=fig)
            else:
                ax_sub = plt.subplot2grid((rows, cols), (row, col), fig=fig, sharex=ax1, sharey=ax1)
            
            posB = np.array([x_pos, 0.0])
            A_translated = translate_polygon(square, posA)
            B_translated = translate_polygon(square, posB)
            sep = separating_distance(A_translated, B_translated)
            
            A_hull = get_convex_hull_vertices(A_translated, closed=True)
            B_hull = get_convex_hull_vertices(B_translated, closed=True)
            
            polygon_A = Polygon(A_hull[:-1], closed=True, facecolor='blue', 
                               alpha=0.4, edgecolor='darkblue', linewidth=1.5)
            ax_sub.add_patch(polygon_A)
            
            polygon_B = Polygon(B_hull[:-1], closed=True, facecolor='red', 
                               alpha=0.4, edgecolor='darkred', linewidth=1.5)
            ax_sub.add_patch(polygon_B)
            
            ax_sub.set_xlim(-1.2, 1.2)
            ax_sub.set_ylim(-0.6, 0.6)
            ax_sub.set_aspect('equal')
            ax_sub.set_title(f'x={x_pos:.2f}\nsep={sep:.4f}', fontsize=9)
            ax_sub.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_separation_sweep.png', dpi=150, bbox_inches='tight')
    print("\nSeparation sweep plot saved to 'test_separation_sweep.png'")
    plt.show()


if __name__ == '__main__':
    print("Testing separating_distance function...")
    print("\n1. Running scenario tests...")
    test_scenarios()
    
    print("\n2. Running distance sweep test...")
    test_distance_sweep()
    
    print("\nTest complete!")

