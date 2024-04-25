import numpy as np


import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import jax

import ott
from ott import problems
from ott.geometry import geometry, pointcloud, epsilon_scheduler
from ott.solvers import linear, quadratic
from ott.solvers.linear import acceleration, sinkhorn
from ott.problems.linear import linear_problem
from ott.tools.sinkhorn_divergence import sinkhorn_divergence

from ott.problems.quadratic import quadratic_problem
from ott.solvers.quadratic import gromov_wasserstein, gromov_wasserstein_lr

def W1(x, y, eps, lse_mode = False):
                    
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
        ott.geometry.pointcloud.PointCloud(x, y, cost_fn=ott.geometry.costs.PNormP(1), epsilon = eps),
        a = a,
        b = b,
        lse_mode=lse_mode,
        min_iterations=0,
        max_iterations=100)
    
    return(ot_solve.reg_ot_cost)

def S1(x, y, eps, lse_mode = False):
                    
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
        ott.geometry.pointcloud.PointCloud(x, y, cost_fn=ott.geometry.costs.PNormP(1), epsilon = eps),
        a = a,
        b = b,
        lse_mode=lse_mode,
        min_iterations=0,
        max_iterations=100)

    ot_solve_xx = linear.solve(
    ott.geometry.pointcloud.PointCloud(x, x, cost_fn=None, epsilon = eps),
    a = a,
    b = a,
    lse_mode=lse_mode,
    min_iterations=0,
    max_iterations=100)
    
    ot_solve_yy = linear.solve(
    ott.geometry.pointcloud.PointCloud(y, y, cost_fn=None, epsilon = eps),
    a = b,
    b = b,
    lse_mode=lse_mode,
    min_iterations=0,
    max_iterations=100)
    return(ot_solve_xy.reg_ot_cost - 0.5 * ot_solve_xx.reg_ot_cost - 0.5 * ot_solve_yy.reg_ot_cost)

def W2(x, y, eps, lse_mode = False):
                        
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
        ott.geometry.pointcloud.PointCloud(x, y, cost_fn=None, epsilon = eps),
        a = a,
        b = b,
        lse_mode=lse_mode,
        min_iterations=0,
        max_iterations=100)

    return(ot_solve.reg_ot_cost)

    
def S2(x, y, eps, lse_mode = False):
                            
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
        ott.geometry.pointcloud.PointCloud(x, y, cost_fn=None, epsilon = eps),
        a = a,
        b = b,
        lse_mode=lse_mode,
        min_iterations=0,
        max_iterations=100)

    ot_solve_xx = linear.solve(
    ott.geometry.pointcloud.PointCloud(x, x, cost_fn=None, epsilon = eps),
    a = a,
    b = a,
    lse_mode=lse_mode,
    min_iterations=0,
    max_iterations=100)
    
    ot_solve_yy = linear.solve(
    ott.geometry.pointcloud.PointCloud(y, y, cost_fn=None, epsilon = eps),
    a = b,
    b = b,
    lse_mode=lse_mode,
    min_iterations=0,
    max_iterations=100)
    return(ot_solve_xy.reg_ot_cost - 0.5 * ot_solve_xx.reg_ot_cost - 0.5 * ot_solve_yy.reg_ot_cost)

def GW(x, y, eps, lse_mode = False):
                                
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
    geom_xx = pointcloud.PointCloud(x=x, y=x).set_scale_cost('max_cost')
    geom_yy = pointcloud.PointCloud(x=y, y=y).set_scale_cost('max_cost')
    
    solver = gromov_wasserstein.GromovWasserstein(epsilon=eps, max_iterations = 100)
    prob = quadratic_problem.QuadraticProblem(geom_xx, geom_yy, a = a, b = b)
    ot_solve = solver(prob)
    
    return(ot_solve.reg_gw_cost)


def GS(x, y, eps, lse_mode = False):
                                    
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
    
    geom_xx = pointcloud.PointCloud(x=x, y=x).set_scale_cost('max_cost')
    geom_yy = pointcloud.PointCloud(x=y, y=y).set_scale_cost('max_cost')
    
    
    solver = gromov_wasserstein.GromovWasserstein(epsilon=eps, max_iterations = 100, lse_mode = lse_mode)
    prob_xy = quadratic_problem.QuadraticProblem(geom_xx, geom_yy, a = a, b = b)
    ot_solve_xy = solver(prob_xy)
    
    solver = gromov_wasserstein.GromovWasserstein(epsilon=eps, max_iterations = 100, lse_mode = lse_mode)
    prob_xx = quadratic_problem.QuadraticProblem(geom_xx, geom_xx, a = a, b = a)
    ot_solve_xx = solver(prob_xx)
     
    solver = gromov_wasserstein.GromovWasserstein(epsilon=eps, max_iterations = 100, lse_mode = lse_mode)
    prob_yy = quadratic_problem.QuadraticProblem(geom_yy, geom_yy, a = b, b = b)
    ot_solve_yy = solver(prob_yy)
    
    return(ot_solve_xy.reg_gw_cost - 0.5 * ot_solve_xx.reg_gw_cost - 0.5 * ot_solve_yy.reg_gw_cost)



def Zeros(x, y, eps, lse_mode = False):
    
    """
    Automatically returns 0, used when Wormhole is trained to only embed and to avoid computational overhead of the decoder


    :param x: (list) list with two elements, the first (x[0]) being the point-cloud coordinates and the second (x[1]) being each points weight)
    :param y: (list) list with two elements, the first (y[0]) being the point-cloud coordinates and the second (y[1]) being each points weight)
    :param eps: (float) coefficient of entropic regularization
    :param lse_mode: (bool) whether to use log-sum-exp mode (if True, more stable for smaller eps, but slower) or kernel mode (default False)
    
    :return zeros: array of zeros
    """ 
    
    return(jnp.zeros([x[0].shape[0]]))
