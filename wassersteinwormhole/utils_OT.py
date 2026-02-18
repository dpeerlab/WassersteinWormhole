import jax
import jax.numpy as jnp
import ott
from ott.geometry import pointcloud, geometry
from ott.problems.quadratic import quadratic_problem
from ott.solvers import linear
from ott.solvers.quadratic import gromov_wasserstein

from tqdm import tqdm
import numpy as np

# --- Helper Functions ---

def _unpack(x_list, y_list):
    """Unpack point cloud and weight lists."""
    return x_list[0], x_list[1], y_list[0], y_list[1]

def _solve_linear(x, y, a, b, eps, lse_mode, num_iter, ot_scale, cost_fn=None):
    """Helper for linear OT problems (W1, W2)."""
    geom = ott.geometry.pointcloud.PointCloud(
        x, y, cost_fn=cost_fn, epsilon=eps, scale_cost=ot_scale
    )
    return linear.solve(
        geom, a=a, b=b, lse_mode=lse_mode, min_iterations=num_iter, max_iterations=num_iter
    )

def _solve_gw(x, y, a, b, eps, lse_mode, num_iter, ot_scale):
    """Helper for Gromov-Wasserstein problems."""
    geom_xx = pointcloud.PointCloud(x=x, y=x, scale_cost=ot_scale)
    geom_yy = pointcloud.PointCloud(x=y, y=y, scale_cost=ot_scale)
    
    # Pass lse_mode to the inner linear solver
    linear_solver = ott.solvers.linear.sinkhorn.Sinkhorn(lse_mode=lse_mode)
    solver = gromov_wasserstein.GromovWasserstein(
        epsilon=eps, max_iterations=num_iter, linear_solver=linear_solver
    )
    prob = quadratic_problem.QuadraticProblem(geom_xx, geom_yy, a=a, b=b)
    return solver(prob)

def _solve_linear_riemannian(x, y, a, b, eps, lse_mode, num_iter, ot_scale, dist_metric):
    """Helper for Riemannian linear OT problems."""
    cost_matrix = dist_metric(x, y)
    geom = geometry.Geometry(
        cost_matrix=cost_matrix, epsilon=eps, scale_cost=ot_scale
    )
    return linear.solve(
        geom, a=a, b=b, lse_mode=lse_mode, min_iterations=num_iter, max_iterations=num_iter
    )

def _solve_gw_riemannian(x, y, a, b, eps, lse_mode, num_iter, ot_scale, dist_metric):
    """Helper for Riemannian Gromov-Wasserstein problems."""
    cost_xx = dist_metric(x, x)
    cost_yy = dist_metric(y, y)
    
    geom_xx = geometry.Geometry(cost_matrix=cost_xx, scale_cost=ot_scale)
    geom_yy = geometry.Geometry(cost_matrix=cost_yy, scale_cost=ot_scale)
    
    linear_solver = ott.solvers.linear.sinkhorn.Sinkhorn(lse_mode=lse_mode)
    solver = gromov_wasserstein.GromovWasserstein(
        epsilon=eps, max_iterations=num_iter, linear_solver=linear_solver
    )
    prob = quadratic_problem.QuadraticProblem(geom_xx, geom_yy, a=a, b=b)
    return solver(prob)

def _calculate_divergence(solve_fn, cost_attr, x, y, a, b, **kwargs):
    """Helper to calculate Sinkhorn Divergence: OT(x,y) - 0.5*OT(x,x) - 0.5*OT(y,y)."""
    ot_xy = solve_fn(x, y, a, b, **kwargs)
    ot_xx = solve_fn(x, x, a, a, **kwargs)
    ot_yy = solve_fn(y, y, b, b, **kwargs)
    return getattr(ot_xy, cost_attr) - 0.5 * getattr(ot_xx, cost_attr) - 0.5 * getattr(ot_yy, cost_attr)

# --- Main Functions ---

def W1(x, y, eps, lse_mode = False, num_iter = 200, ot_scale = 1):
    """Calculate W1 (EMD) between two weighted point clouds"""
    x_pts, a, y_pts, b = _unpack(x, y)
    cost_fn = ott.geometry.costs.PNormP(1)
    out = _solve_linear(x_pts, y_pts, a, b, eps, lse_mode, num_iter, ot_scale, cost_fn)
    return out.reg_ot_cost

def S1(x, y, eps, lse_mode = False, num_iter = 200, ot_scale = 1):
    """Calculate EMD Sinkhorn divergence between two weighted point clouds"""
    x_pts, a, y_pts, b = _unpack(x, y)
    cost_fn = ott.geometry.costs.PNormP(1)
    return _calculate_divergence(
        _solve_linear, 'reg_ot_cost', x_pts, y_pts, a, b,
        eps=eps, lse_mode=lse_mode, num_iter=num_iter, ot_scale=ot_scale, cost_fn=cost_fn
    )

def W2(x, y, eps, lse_mode = False, num_iter = 200, ot_scale = 1):
    """Calculate W2 between two weighted point clouds"""
    x_pts, a, y_pts, b = _unpack(x, y)
    out = _solve_linear(x_pts, y_pts, a, b, eps, lse_mode, num_iter, ot_scale, cost_fn=None)
    return out.reg_ot_cost

def S2(x, y, eps, lse_mode = False, num_iter = 200, ot_scale = 1):
    """Calculate Sinkhorn Divergnece (S2) between two weighted point clouds"""
    x_pts, a, y_pts, b = _unpack(x, y)
    return _calculate_divergence(
        _solve_linear, 'reg_ot_cost', x_pts, y_pts, a, b,
        eps=eps, lse_mode=lse_mode, num_iter=num_iter, ot_scale=ot_scale, cost_fn=None
    )

def GW(x, y, eps, lse_mode = False, num_iter = 200, ot_scale = 1):
    """Calculate Gromov-Wasserstein distance between two weighted point clouds"""
    x_pts, a, y_pts, b = _unpack(x, y)
    out = _solve_gw(x_pts, y_pts, a, b, eps, lse_mode, num_iter, ot_scale)
    return out.reg_gw_cost

def GS(x, y, eps, lse_mode = False, num_iter = 200, ot_scale = 1):
    """Calculate Gromov-Wasserstein based Sinkhorn Divergence (GS) distance between two weighted point clouds"""
    x_pts, a, y_pts, b = _unpack(x, y)
    return _calculate_divergence(
        _solve_gw, 'reg_gw_cost', x_pts, y_pts, a, b,
        eps=eps, lse_mode=lse_mode, num_iter=num_iter, ot_scale=ot_scale
    )

def W_R(x, y, eps, lse_mode=False, num_iter=200, ot_scale=1, dist_metric=None):
    """Calculate Riemannian Wasserstein distance using a custom distance metric."""
    if dist_metric is None:
        raise ValueError("dist_metric must be provided for Riemannian OT")
    x_pts, a, y_pts, b = _unpack(x, y)
    out = _solve_linear_riemannian(x_pts, y_pts, a, b, eps, lse_mode, num_iter, ot_scale, dist_metric)
    return out.reg_ot_cost

def S_R(x, y, eps, lse_mode=False, num_iter=200, ot_scale=1, dist_metric=None):
    """Calculate Riemannian Sinkhorn Divergence using a custom distance metric."""
    if dist_metric is None:
        raise ValueError("dist_metric must be provided for Riemannian OT")
    x_pts, a, y_pts, b = _unpack(x, y)
    return _calculate_divergence(
        _solve_linear_riemannian, 'reg_ot_cost', x_pts, y_pts, a, b,
        eps=eps, lse_mode=lse_mode, num_iter=num_iter, ot_scale=ot_scale, dist_metric=dist_metric
    )

def GW_R(x, y, eps, lse_mode=False, num_iter=200, ot_scale=1, dist_metric=None):
    """Calculate Riemannian Gromov-Wasserstein distance using a custom distance metric."""
    if dist_metric is None:
        raise ValueError("dist_metric must be provided for Riemannian OT")
    x_pts, a, y_pts, b = _unpack(x, y)
    out = _solve_gw_riemannian(x_pts, y_pts, a, b, eps, lse_mode, num_iter, ot_scale, dist_metric)
    return out.reg_gw_cost

def GS_R(x, y, eps, lse_mode=False, num_iter=200, ot_scale=1, dist_metric=None):
    """Calculate Riemannian Gromov-Sinkhorn Divergence using a custom distance metric."""
    if dist_metric is None:
        raise ValueError("dist_metric must be provided for Riemannian OT")
    x_pts, a, y_pts, b = _unpack(x, y)
    return _calculate_divergence(
        _solve_gw_riemannian, 'reg_gw_cost', x_pts, y_pts, a, b,
        eps=eps, lse_mode=lse_mode, num_iter=num_iter, ot_scale=ot_scale, dist_metric=dist_metric
    )

def Zeros(x, y, eps, lse_mode = False, num_iter = 200, ot_scale = 1):
    """Automatically returns 0, used when Wormhole is trained to only embed and to avoid computational overhead of the decoder"""
    return 0

def auto_find_num_iter(point_clouds, weights, eps, lse_mode, ot_scale, ot_func, num_calc=100, sample_size=2048, dist_metric=None):
    """
    Find the minimum number of iterations for which at least 95% of OT calculations converge.
    Uses jax.vmap to run calculations in parallel and inspects error traces.
    """
    
    if ot_func == 'Zeros':
        return 100

    # 1. Prepare Batch Data
    num_clouds = len(point_clouds)
    idx1 = np.random.choice(num_clouds, num_calc)
    idx2 = np.random.choice(num_clouds, num_calc)
    
    batch_x, batch_y, batch_a, batch_b = [], [], [], []
    dim = point_clouds[0].shape[1]

    print(f"Preparing batch of {num_calc} pairs for {ot_func} convergence check...")
    for i in range(num_calc):
        x, a = point_clouds[idx1[i]], weights[idx1[i]]
        y, b = point_clouds[idx2[i]], weights[idx2[i]]
        
        # Subsample or Pad X
        if len(x) > sample_size:
            idx = np.random.choice(len(x), sample_size, replace=False)
            x, a = x[idx], a[idx]
        elif len(x) < sample_size:
            pad_size = sample_size - len(x)
            x = np.concatenate([x, np.zeros((pad_size, dim))])
            a = np.concatenate([a, np.zeros(pad_size)])
            
        # Subsample or Pad Y
        if len(y) > sample_size:
            idx = np.random.choice(len(y), sample_size, replace=False)
            y, b = y[idx], b[idx]
        elif len(y) < sample_size:
            pad_size = sample_size - len(y)
            y = np.concatenate([y, np.zeros((pad_size, dim))])
            b = np.concatenate([b, np.zeros(pad_size)])
            
        # Normalize weights
        a = a / np.sum(a)
        b = b / np.sum(b)
        
        batch_x.append(x)
        batch_y.append(y)
        batch_a.append(a)
        batch_b.append(b)

    bx = jnp.array(batch_x)
    by = jnp.array(batch_y)
    ba = jnp.array(batch_a)
    bb = jnp.array(batch_b)

    # 2. Define Solver
    max_iter = 5000
    
    def solve_single(x, y, a, b):
        if '_R' in ot_func:
            if dist_metric is None:
                 # Fallback or error, but we should assume it's provided if _R is used
                 return jnp.zeros((max_iter // 10,))
            
            if 'W' in ot_func or 'S' in ot_func:
                cost_matrix = dist_metric(x, y)
                geom = geometry.Geometry(
                    cost_matrix=cost_matrix, epsilon=eps, scale_cost=ot_scale
                )
                out = linear.solve(
                    geom, a=a, b=b, lse_mode=lse_mode, min_iterations=0, max_iterations=max_iter
                )

                return out.errors
            elif 'G' in ot_func:
                cost_xx = dist_metric(x, x)
                cost_yy = dist_metric(y, y)
                
                geom_xx = geometry.Geometry(cost_matrix=cost_xx, scale_cost=ot_scale)
                geom_yy = geometry.Geometry(cost_matrix=cost_yy, scale_cost=ot_scale)
                
                linear_solver = ott.solvers.linear.sinkhorn.Sinkhorn(lse_mode=lse_mode)
                solver = gromov_wasserstein.GromovWasserstein(
                    epsilon=eps, max_iterations=max_iter, linear_solver=linear_solver
                )
                prob = quadratic_problem.QuadraticProblem(geom_xx, geom_yy, a=a, b=b)
                out = solver(prob)
                return out.errors

        if 'W' in ot_func or 'S' in ot_func:
            cost_fn = ott.geometry.costs.PNormP(1) if '1' in ot_func else None
            geom = ott.geometry.pointcloud.PointCloud(x, y, cost_fn=cost_fn, epsilon=eps, scale_cost=ot_scale)
            # We use min_iterations=0 to allow early stopping and recording -1 in errors
            out = linear.solve(geom, a=a, b=b, lse_mode=lse_mode, min_iterations=0, max_iterations=max_iter)
            return out.errors
        elif 'G' in ot_func:
            geom_xx = pointcloud.PointCloud(x=x, y=x, scale_cost=ot_scale)
            geom_yy = pointcloud.PointCloud(x=y, y=y, scale_cost=ot_scale)
            linear_solver = ott.solvers.linear.sinkhorn.Sinkhorn(lse_mode=lse_mode)
            solver = gromov_wasserstein.GromovWasserstein(
                epsilon=eps, max_iterations=max_iter, linear_solver=linear_solver, min_iterations=0
            )
            prob = quadratic_problem.QuadraticProblem(geom_xx, geom_yy, a=a, b=b)
            out = solver(prob)
            return out.errors
        return jnp.zeros((max_iter // 10,)) # Fallback

    # 3. Run Vmapped Solver
    print(f"Running {ot_func} solver for {max_iter} iterations on {num_calc} pairs...")
    batch_errors = jax.jit(jax.vmap(solve_single))(bx, by, ba, bb)
    
    # 4. Analyze Convergence
    # errors is (num_calc, max_iter // inner_iter)
    # -1 indicates the solver stopped (converged) before this step.
    
    converged_fraction = jnp.mean(batch_errors == -1.0, axis=0)
    
    condition = converged_fraction >= 0.95
    if not jnp.any(condition):
        print(f"Did not reach 95% convergence within {max_iter} iterations.")
        return max_iter
        
    first_converged_idx = jnp.argmax(condition)
    
    # Calculate iterations. 
    # If index k is -1, it took at most k * 10 iterations.
    recommended_iter = int((first_converged_idx) * 10)
    if recommended_iter == 0: recommended_iter = 100 
    
    # Round up to nearest 100
    recommended_iter = int(np.ceil(recommended_iter / 100.0)) * 100
    
    print(f"95% convergence reached at approx {recommended_iter} iterations.")
    return recommended_iter






