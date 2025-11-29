"""
Example: place movable circular objects (fixed radii and fixed orientations irrelevant) to minimize
sum of squared distances to target points, subject to non-overlap among movables and with fixed obstacles.

Solver: SciPy's SLSQP (nonlinear constrained optimizer). Uses multiple random restarts to improve chances
of finding a good feasible solution.

How to use: pip install numpy scipy matplotlib
Then run: python circle_placement_scipy.py

This script contains:
- model data (example)
- objective and constraint functions
- multi-start optimization
- simple plot of final layout
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ---------------------- Problem data (example) ----------------------
# Movable circles: list of dicts with 'r' (radius) and 'target' (x,y)
movables = [
    {'r': 0.5, 'target': np.array([0.0, 0.0])},
    {'r': 0.4, 'target': np.array([1.0, 0.0])},
    {'r': 0.45, 'target': np.array([0.5, 0.8])},
]

# Fixed obstacles: list of dicts with 'r' and 'c' (center)
fixed_obstacles = [
    {'r': 0.6, 'c': np.array([2.0, 0.5])},
    {'r': 0.5, 'c': np.array([-1.2, -0.3])},
]

# Optional placement bounds (for initial guesses and to keep variables reasonable)
# Bounds are ((xmin,xmax), (ymin,ymax)) used for random restarts and optional variable bounds in optimizer
placement_bounds = ((-3.0, 3.0), (-3.0, 3.0))

# Solver/algorithm parameters
NUM_RESTARTS = 30
MAXITER = 400
TOL = 1e-6

# ---------------------- Helper functions ----------------------

def pack_xy(xy_list):
    """Convert list of (x,y) pairs to flat vector and back"""
    return np.hstack([np.asarray(xy).reshape(-1) for xy in xy_list])


def unpack_xy(xvec):
    """Convert flat vector to list of 2D points"""
    pts = xvec.reshape(-1, 2)
    return [pts[i] for i in range(pts.shape[0])]


# ---------------------- Objective ----------------------

def objective(xvec):
    """Sum of squared distances to targets (simple quadratic objective)."""
    pts = unpack_xy(xvec)
    val = 0.0
    for i, p in enumerate(pts):
        t = movables[i]['target']
        val += np.sum((p - t) ** 2)
    return val


# ---------------------- Constraints ----------------------

def pairwise_nonoverlap_constraints(xvec):
    """Return array of non-overlap constraint values (should be >= 0 when satisfied).

    For movable i and k: g_ik(x) = dist^2 - (ri + rk)^2 >= 0
    For movable i and fixed obstacle j: g_ij(x) = dist^2 - (ri + sj)^2 >= 0
    """
    pts = unpack_xy(xvec)
    constr = []

    # movable-movable
    n = len(pts)
    for i in range(n):
        for k in range(i + 1, n):
            d2 = np.sum((pts[i] - pts[k]) ** 2)
            mind2 = (movables[i]['r'] + movables[k]['r']) ** 2
            constr.append(d2 - mind2)

    # movable-fixed
    for i in range(n):
        for j, obs in enumerate(fixed_obstacles):
            d2 = np.sum((pts[i] - obs['c']) ** 2)
            mind2 = (movables[i]['r'] + obs['r']) ** 2
            constr.append(d2 - mind2)

    return np.array(constr)


def make_scipy_constraints():
    """Return list of constraint dicts for scipy.optimize.minimize (inequalities g(x) >= 0)."""

    def con_fun(x):
        return pairwise_nonoverlap_constraints(x)

    # SciPy expects constraints either as dict or sequence; we'll create one dict per inequality.
    g_vals = pairwise_nonoverlap_constraints(np.zeros(len(movables) * 2))
    num_constraints = len(g_vals)

    constraints = []
    for idx in range(num_constraints):
        constraints.append({
            'type': 'ineq',
            'fun': (lambda idx: (lambda x: con_fun(x)[idx]))(idx)
        })
    return constraints


# ---------------------- Optimization routine ----------------------

def optimize_with_restarts(num_restarts=20):
    n = len(movables)
    dim = 2 * n
    best_res = None

    constraints = make_scipy_constraints()

    for restart in range(num_restarts):
        # Random initial guess within bounds near targets
        x0 = np.zeros(dim)
        for i in range(n):
            tx, ty = movables[i]['target']
            # jitter around target, scaled by radii
            jitter_scale = 0.5
            x0[2 * i:2 * i + 2] = np.array([tx, ty]) + jitter_scale * (np.random.rand(2) - 0.5)

        # Optionally, project initial guess to respect bounds
        for i in range(n):
            x0[2 * i] = np.clip(x0[2 * i], placement_bounds[0][0], placement_bounds[0][1])
            x0[2 * i + 1] = np.clip(x0[2 * i + 1], placement_bounds[1][0], placement_bounds[1][1])

        res = minimize(
            objective,
            x0,
            method='SLSQP',
            constraints=constraints,
            options={'maxiter': MAXITER, 'ftol': TOL, 'disp': False}
        )

        # Evaluate feasibility (all constraints >= -tol)
        g = pairwise_nonoverlap_constraints(res.x)
        feasible = np.all(g >= -1e-6)

        if feasible:
            if best_res is None or res.fun < best_res.fun:
                best_res = res
        else:
            # If not feasible, still consider if it's the best by penalizing constraint violation
            penalty = np.sum(np.minimum(0, g) ** 2) * 1e6
            penalized_obj = res.fun + penalty
            if best_res is None or penalized_obj < (best_res.fun if best_res is not None else np.inf):
                # store but mark as infeasible (we'll prefer feasible solutions when available)
                res.penalized_obj = penalized_obj
                res.feasible = False
                best_res = res

    return best_res


# ---------------------- Run and visualize ----------------------

def plot_solution(xvec, title='Placement result'):
    pts = unpack_xy(xvec)
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    # plot fixed obstacles
    for obs in fixed_obstacles:
        circle = plt.Circle(obs['c'], obs['r'], fill=True, alpha=0.3, label='fixed')
        ax.add_patch(circle)

    # plot movables
    for i, p in enumerate(pts):
        r = movables[i]['r']
        targ = movables[i]['target']
        circ = plt.Circle(p, r, fill=False, linewidth=2)
        ax.add_patch(circ)
        ax.plot([p[0], targ[0]], [p[1], targ[1]], '--', linewidth=1)
        ax.scatter([targ[0]], [targ[1]], marker='x')
        ax.text(p[0] + 0.02, p[1] + 0.02, f'i={i}')

    # plot settings
    ax.set_xlim(placement_bounds[0])
    ax.set_ylim(placement_bounds[1])
    ax.set_title(title)
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    np.random.seed(0)
    res = optimize_with_restarts(NUM_RESTARTS)

    if res is None:
        print('No result found.')
    else:
        print('Optimization result:')
        print('  success:', getattr(res, 'success', None))
        print('  fun (objective):', res.fun)
        if hasattr(res, 'penalized_obj'):
            print('  penalized_obj (for infeasible stored result):', res.penalized_obj)
        print('  x:', res.x)

        # Check constraint values
        g = pairwise_nonoverlap_constraints(res.x)
        print('  minimum constraint value:', np.min(g))

        # plot
        plot_solution(res.x)

    print('\nNotes:')
    print('- This is a local optimizer; multi-starts improve chances of finding a feasible/good solution.')
    print("- For strict combinatorial constraints (e.g., rectangles with discrete orientations),")
    print('  consider a MIP formulation (MILP) with binary variables and a solver like Gurobi/CPLEX/SCIP or OR-Tools CP-SAT.')
