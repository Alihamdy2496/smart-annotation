# Smart Annotation Placement

This project implements a high-performance, automated system for placing text annotations (or any 2D polygonal shapes) in a layout such that they do not overlap with each other or with fixed obstacles, while remaining as close as possible to their target positions.

## The Problem

In automated drafting, mapping, and data visualization, placing labels is a classic problem. The goal is to:
1.  **Avoid Overlaps**: No two movable objects (annotations) can overlap.
2.  **Avoid Obstacles**: Movable objects cannot overlap with fixed obstacles (e.g., geometry lines, other components).
3.  **Maintain Proximity**: Each object has a "target" position (its ideal location) and should be moved as little as possible.

This is a constrained optimization problem that becomes computationally expensive as the number of objects increases ($O(N^2)$ interactions).

## The Solution

We use an **Iterative Projection Algorithm** combined with **Spatial Indexing** and **Hybrid Acceleration**.

### Core Algorithm
The system uses a physics-inspired relaxation approach:
1.  **Collision Detection**: We use the **Separating Axis Theorem (SAT)** to detect overlaps between convex polygons.
2.  **Projection**: When an overlap is detected, we calculate the minimum translation vector (MTV) required to separate the objects.
3.  **Iterative Solver**: We apply these translation vectors iteratively. In each step, objects are "pushed" away from overlaps. This repeats until convergence (zero overlaps) or a maximum iteration count is reached.

### Performance Optimizations

To handle hundreds of objects efficiently, we implemented several key optimizations:

#### 1. Spatial Indexing (KD-Tree)
Instead of checking every object against every other object ($O(N^2)$), we use a `cKDTree` (from `scipy.spatial`) to find potential collision candidates within a search radius. This reduces the complexity to approximately $O(N \log N)$.

#### 2. Hybrid Numba + Python Architecture
This is the core of our performance strategy. We use a "Fast Path / Robust Fallback" pattern:

*   **Fast Path (Numba)**:
    *   We use **Numba** to JIT-compile the collision detection and resolution logic into optimized machine code.
    *   For the majority of cases (single movable vs. multiple fixed obstacles), the Numba path runs orders of magnitude faster than Python.
    *   It uses a "Greedy" resolution strategy with random perturbations to escape local minima.

*   **Robust Fallback (Python)**:
    *   In complex scenarios (e.g., an object trapped between multiple obstacles), the greedy Numba solver might fail to resolve all overlaps within the iteration limit.
    *   The system detects this failure and automatically **falls back** to a robust, pure Python implementation for that specific batch.
    *   This ensures **100% reliability** (zero overlaps) without sacrificing the speed benefits of Numba for the 95% of "easy" cases.

### Staged Processing
The optimization runs in three stages to balance speed and quality:
1.  **Stage 1 (Global Projection)**: Moves objects to approximate non-overlapping positions.
2.  **Stage 2 (Batch Refinement)**: Processes objects in batches using the **Hybrid Numba/Python** engine with tighter constraints.
3.  **Stage 3 (Final Tightening)**: A final pass to ensure strict adherence to the minimum separation distance.

## Project Structure

*   `main.py`: Entry point. Loads data, runs optimization, and saves results.
*   `optimizer.py`: Core optimization logic, including the hybrid Numba/Python solver.
*   `numba_utils.py`: JIT-compiled functions for high-speed collision detection.
*   `polygon_utils.py`: Robust Python geometry functions (SAT, convex hulls).
*   `json_helper.py`: Handles JSON I/O and data conversion.
*   `plotting.py`: Visualization utilities.

## Usage

Run the main script:

```bash
python3 main.py
```

This will:
1.  Load problem data from `AnnotationCleaner_CurveLoops.json`.
2.  Run the optimization.
3.  Save the result to `output.json`.
4.  Generate visualization images: `before.png` and `after.png`.