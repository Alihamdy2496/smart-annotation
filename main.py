"""
Main execution script for object placement optimization.

This script reads problem data (movable objects and fixed obstacles),
runs the optimization, and visualizes the results.
"""

import numpy as np
from optimizer import optimize, calculate_displacement_metric
from json_helper import load_problem_data, save_optimized_output
from plotting import plot_result
import time

if __name__ == "__main__":
    np.random.seed(0)

    # Load problem data from JSON
    json_path = "AnnotationCleaner_CurveLoops.json"
    movables, fixed_obstacles, placement_bounds = load_problem_data(json_path)

    print(
        f"Computed placement bounds: X={placement_bounds[0]}, Y={placement_bounds[1]}"
    )

    # Run optimization
    start_time = time.time()
    result, x0 = optimize(movables, fixed_obstacles)
    end_time = time.time()
    print(
        f"Optimization completed in {end_time - start_time:.2f} seconds, {(end_time - start_time) / 60:.2f} minutes"
    )

    # Calculate and print displacement metric
    displacement_metric = calculate_displacement_metric(x0, result)
    print(f"Displacement Preservation Metric: {displacement_metric:.4f}")

    # Save optimized output
    save_optimized_output(result, movables, output_path="output.json")

    # Visualize results
    plot_result(result, x0, movables, fixed_obstacles, placement_bounds)
