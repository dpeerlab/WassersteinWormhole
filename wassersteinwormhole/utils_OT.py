

import ott
from ott.geometry import pointcloud
from ott.problems.quadratic import quadratic_problem
from ott.solvers import linear
from ott.solvers.quadratic import gromov_wasserstein

from tqdm import tqdm
import numpy as np

def W1(x, y, eps, lse_mode = False, num_iter = 200, ot_scale = 1):
                    
    """
    Calculate W1 (EMD) between two weighted point clouds


    :param x: (list) list with two elements, the first (x[0]) being the point-cloud coordinates and the second (x[1]) being each points weight)
    :param y: (list) list with two elements, the first (y[0]) being the point-cloud coordinates and the second (y[1]) being each points weight)
    :param eps: (float) coefficient of entropic regularization
    :param lse_mode: (bool) whether to use log-sum-exp mode (if True, more stable for smaller eps, but slower) or kernel mode (default False)
    
    :return W1: EMD distance between x and y 
    """ 

    x,a = x[0], x[1]
    y,b = y[0], y[1]
    ot_solve = linear.solve(
        ott.geometry.pointcloud.PointCloud(x, y, cost_fn=ott.geometry.costs.PNormP(1), epsilon = eps, scale_cost = ot_scale),
        a = a,
        b = b,
        lse_mode=lse_mode,
        min_iterations = num_iter,
        max_iterations = num_iter)
    
    return(ot_solve.reg_ot_cost)

def S1(x, y, eps, lse_mode = False, num_iter = 200, ot_scale = 1):
                    
    """
    Calculate EMD Sinkhorn divergence between two weighted point clouds


    :param x: (list) list with two elements, the first (x[0]) being the point-cloud coordinates and the second (x[1]) being each points weight)
    :param y: (list) list with two elements, the first (y[0]) being the point-cloud coordinates and the second (y[1]) being each points weight)
    :param eps: (float) coefficient of entropic regularization
    :param lse_mode: (bool) whether to use log-sum-exp mode (if True, more stable for smaller eps, but slower) or kernel mode (default False)
    
    :return S1: L1 Sinkhorn divergence between x and y 
    """ 

    x,a = x[0], x[1]
    y,b = y[0], y[1]
    ot_solve_xy = linear.solve(
        ott.geometry.pointcloud.PointCloud(x, y, cost_fn=ott.geometry.costs.PNormP(1), epsilon = eps, scale_cost = ot_scale),
        a = a,
        b = b,
        lse_mode=lse_mode,
        min_iterations = num_iter,
        max_iterations = num_iter)

    ot_solve_xx = linear.solve(
    ott.geometry.pointcloud.PointCloud(x, x, cost_fn=None, epsilon = eps, scale_cost = ot_scale),
    a = a,
    b = a,
    lse_mode=lse_mode,
    min_iterations = num_iter,
    max_iterations = num_iter)
    
    ot_solve_yy = linear.solve(
    ott.geometry.pointcloud.PointCloud(y, y, cost_fn=None, epsilon = eps, scale_cost = ot_scale),
    a = b,
    b = b,
    lse_mode=lse_mode,
    min_iterations = num_iter,
    max_iterations = num_iter)
    return(ot_solve_xy.reg_ot_cost - 0.5 * ot_solve_xx.reg_ot_cost - 0.5 * ot_solve_yy.reg_ot_cost)

def W2(x, y, eps, lse_mode = False, num_iter = 200, ot_scale = 1):
                        
    """
    Calculate W2 between two weighted point clouds


    :param x: (list) list with two elements, the first (x[0]) being the point-cloud coordinates and the second (x[1]) being each points weight)
    :param y: (list) list with two elements, the first (y[0]) being the point-cloud coordinates and the second (y[1]) being each points weight)
    :param eps: (float) coefficient of entropic regularization
    :param lse_mode: (bool) whether to use log-sum-exp mode (if True, more stable for smaller eps, but slower) or kernel mode (default False)
    
    :return W2: Wasserstien distance between x and y 
    """ 

    x,a = x[0], x[1]
    y,b = y[0], y[1]
    
        
    ot_solve = linear.solve(
        ott.geometry.pointcloud.PointCloud(x, y, cost_fn=None, epsilon = eps, scale_cost = ot_scale),
        a = a,
        b = b,
        lse_mode=lse_mode,
        min_iterations = num_iter,
        max_iterations = num_iter)

    return(ot_solve.reg_ot_cost)

    
def S2(x, y, eps, lse_mode = False, num_iter = 200, ot_scale = 1):
                            
    """
    Calculate Sinkhorn Divergnece (S2) between two weighted point clouds


    :param x: (list) list with two elements, the first (x[0]) being the point-cloud coordinates and the second (x[1]) being each points weight)
    :param y: (list) list with two elements, the first (y[0]) being the point-cloud coordinates and the second (y[1]) being each points weight)
    :param eps: (float) coefficient of entropic regularization
    :param lse_mode: (bool) whether to use log-sum-exp mode (if True, more stable for smaller eps, but slower) or kernel mode (default False)
    
    :return S2: Sinkhorn Divergnece between x and y 
    """ 

    x,a = x[0], x[1]
    y,b = y[0], y[1]
        
    ot_solve_xy = linear.solve(
        ott.geometry.pointcloud.PointCloud(x, y, cost_fn=None, epsilon = eps, scale_cost = ot_scale),
        a = a,
        b = b,
        lse_mode=lse_mode,
        min_iterations = num_iter,
        max_iterations = num_iter)

    ot_solve_xx = linear.solve(
    ott.geometry.pointcloud.PointCloud(x, x, cost_fn=None, epsilon = eps, scale_cost = ot_scale),
    a = a,
    b = a,
    lse_mode=lse_mode,
    min_iterations = num_iter,
    max_iterations = num_iter)
    
    ot_solve_yy = linear.solve(
    ott.geometry.pointcloud.PointCloud(y, y, cost_fn=None, epsilon = eps, scale_cost = ot_scale),
    a = b,
    b = b,
    lse_mode=lse_mode,
    min_iterations = num_iter,
    max_iterations = num_iter)
    return(ot_solve_xy.reg_ot_cost - 0.5 * ot_solve_xx.reg_ot_cost - 0.5 * ot_solve_yy.reg_ot_cost)

def GW(x, y, eps, lse_mode = False, num_iter = 200, ot_scale = 1):
                                
    """
    Calculate Gromov-Wasserstein distance between two weighted point clouds


    :param x: (list) list with two elements, the first (x[0]) being the point-cloud coordinates and the second (x[1]) being each points weight)
    :param y: (list) list with two elements, the first (y[0]) being the point-cloud coordinates and the second (y[1]) being each points weight)
    :param eps: (float) coefficient of entropic regularization
    :param lse_mode: (bool) whether to use log-sum-exp mode (if True, more stable for smaller eps, but slower) or kernel mode (default False)
    
    :return GW: GW distance between x and y 
    """ 

    x,a = x[0], x[1]
    y,b = y[0], y[1]
    
    geom_xx = pointcloud.PointCloud(x=x, y=x, scale_cost=ot_scale)
    geom_yy = pointcloud.PointCloud(x=y, y=y, scale_cost=ot_scale)
    
    solver = gromov_wasserstein.GromovWasserstein(epsilon=eps, max_iterations = num_iter, linear_solver = ott.solvers.linear.sinkhorn.Sinkhorn())
    prob = quadratic_problem.QuadraticProblem(geom_xx, geom_yy, a = a, b = b)
    ot_solve = solver(prob)
    
    return(ot_solve.reg_gw_cost)


def GS(x, y, eps, lse_mode = False, num_iter = 200, ot_scale = 1):
                                    
    """
    Calculate Gromov-Wasserstein based Sinkhorn Divergence (GS) distance between two weighted point clouds


    :param x: (list) list with two elements, the first (x[0]) being the point-cloud coordinates and the second (x[1]) being each points weight)
    :param y: (list) list with two elements, the first (y[0]) being the point-cloud coordinates and the second (y[1]) being each points weight)
    :param eps: (float) coefficient of entropic regularization
    :param lse_mode: (bool) whether to use log-sum-exp mode (if True, more stable for smaller eps, but slower) or kernel mode (default False)
    
    :return GS: Gromov-Wasserstein Sinkhorn Divergence between x and y 
    """ 

    x,a = x[0], x[1]
    y,b = y[0], y[1]
    
        
    geom_xx = pointcloud.PointCloud(x=x, y=x, scale_cost=ot_scale)
    geom_yy = pointcloud.PointCloud(x=y, y=y, scale_cost=ot_scale)
        
    solver = gromov_wasserstein.GromovWasserstein(epsilon=eps, max_iterations = num_iter,  linear_solver = ott.solvers.linear.sinkhorn.Sinkhorn())
    prob_xy = quadratic_problem.QuadraticProblem(geom_xx, geom_yy, a = a, b = b)
    ot_solve_xy = solver(prob_xy)

    solver = gromov_wasserstein.GromovWasserstein(epsilon=eps, max_iterations = num_iter, linear_solver = ott.solvers.linear.sinkhorn.Sinkhorn())
    prob_xx = quadratic_problem.QuadraticProblem(geom_xx, geom_xx, a = a, b = a)
    ot_solve_xx = solver(prob_xx)
        
    solver = gromov_wasserstein.GromovWasserstein(epsilon=eps, max_iterations = num_iter,linear_solver = ott.solvers.linear.sinkhorn.Sinkhorn())
    prob_yy = quadratic_problem.QuadraticProblem(geom_yy, geom_yy, a = b, b = b)
    ot_solve_yy = solver(prob_yy)
    
    return(ot_solve_xy.reg_gw_cost - 0.5 * ot_solve_xx.reg_gw_cost - 0.5 * ot_solve_yy.reg_gw_cost)

def Zeros(x, y, eps, lse_mode = False, num_iter = 200, ot_scale = 1):
    
    """
    Automatically returns 0, used when Wormhole is trained to only embed and to avoid computational overhead of the decoder


    :param x: (list) list with two elements, the first (x[0]) being the point-cloud coordinates and the second (x[1]) being each points weight)
    :param y: (list) list with two elements, the first (y[0]) being the point-cloud coordinates and the second (y[1]) being each points weight)
    :param eps: (float) coefficient of entropic regularization
    :param lse_mode: (bool) whether to use log-sum-exp mode (if True, more stable for smaller eps, but slower) or kernel mode (default False)
    
    :return zeros: array of zeros
    """ 
    
    return(0)

def auto_find_num_iter(point_clouds, weights, eps, lse_mode, ot_scale, ot_func, num_calc=100, sample_size=2048):
    """
    Find the minimum number of iterations for which at least 80% of OT calculations converge.

    It tests a predefined list of iteration counts on randomly sampled pairs of point clouds.
    For performance, if a point cloud contains more points than `sample_size`, a random subset
    is used for the calculation.

    :param point_clouds: (list) A list of point cloud coordinate arrays.
    :param weights: (list) A list of corresponding weight arrays for each point cloud.
    :param eps: (float) Coefficient of entropic regularization.
    :param lse_mode: (bool) Whether to use log-sum-exp mode.
    :param ot_scale: (float) Scaling factor for the cost matrix.
    :param ot_func: (str) The name of the OT function to test (e.g., 'W1', 'GW').
    :param num_calc: (int) The number of random pairs to test for each iteration count.
    :param sample_size: (int) The number of points to sample from larger point clouds.

    :return: (int) The recommended number of iterations.
    """
    
    # Handle the 'Zeros' case which involves no solver
    if ot_func == 'Zeros':
        return 100

    num_iter_test = [100, 200, 500, 1000, 5000]
    num_clouds = len(point_clouds)

    for n_iter in num_iter_test:
        converged_count = 0
        for _ in tqdm(range(num_calc), desc=f"Testing {ot_func} convergence with {n_iter} iterations"):
            # Randomly select two different point clouds for comparison
            idx1, idx2 = np.random.choice(num_clouds, 2, replace=False)
            x_points, a_weights = point_clouds[idx1], weights[idx1]
            y_points, b_weights = point_clouds[idx2], weights[idx2]

            # --- OPTIMIZATION: Sample large point clouds to speed up calculation ---
            if len(x_points) > sample_size:
                indices_x = np.random.choice(len(x_points), sample_size, replace=False)
                x_points = x_points[indices_x]
                a_weights = a_weights[indices_x]
            
            if len(y_points) > sample_size:
                indices_y = np.random.choice(len(y_points), sample_size, replace=False)
                y_points = y_points[indices_y]
                b_weights = b_weights[indices_y]
            # --------------------------------------------------------------------

            # Replicate solver logic to access the convergence status
            ot_solve = None
            try:
                if 'W' in ot_func or 'S' in ot_func:
                    # Logic for Wasserstein, Sinkhorn, and their divergences
                    cost_function = ott.geometry.costs.PNormP(1) if '1' in ot_func else None
                    geom = pointcloud.PointCloud(
                        x_points, y_points, 
                        cost_fn=cost_function, 
                        epsilon=eps,
                        scale_cost=ot_scale
                    )
                    ot_solve = linear.solve(
                        geom, a=a_weights, b=b_weights,
                        lse_mode=lse_mode,
                        min_iterations=n_iter,
                        max_iterations=n_iter
                    )
                elif 'G' in ot_func:
                    # Logic for Gromov-Wasserstein and its divergence
                    geom_xx = pointcloud.PointCloud(x=x_points, y=x_points, scale_cost=ot_scale)
                    geom_yy = pointcloud.PointCloud(x=y_points, y=y_points, scale_cost=ot_scale)
                    
                    prob = quadratic_problem.QuadraticProblem(
                        geom_xx, geom_yy, a=a_weights, b=b_weights
                    )
                    solver = gromov_wasserstein.GromovWasserstein(
                        epsilon=eps, lse_mode=lse_mode, max_iterations=n_iter
                    )
                    ot_solve = solver(prob)

                if ot_solve and ot_solve.converged:
                    converged_count += 1
            
            except Exception as e:
                # In case of numerical errors, count as not converged
                print(f"Solver failed for {ot_func} with {n_iter} iterations: {e}")
                continue

        # Check if the convergence rate is 80% or higher
        convergence_rate = round((converged_count / num_calc) * 100)
        print(f"Convergence rate for {ot_func} with {n_iter} iterations: {convergence_rate}%")

        if (converged_count / num_calc) >= 0.8:
            return n_iter

    # If no value meets the criterion, return the highest tested number of iterations
    return num_iter_test[-1]






