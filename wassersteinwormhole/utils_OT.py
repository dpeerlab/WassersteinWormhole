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
from ott.tools.sinkhorn_divergence import sinkhorn_divergence


def W1(x, y, eps, lse_mode = False):
    x,a = x[0], x[1]
    y,b = y[0], y[1]
    ot_solve = linear.solve(
        ott.geometry.pointcloud.PointCloud(x, y, cost_fn=ott.geometry.costs.PNormP(1), epsilon = eps),
        a = a,
        b = b,
        lse_mode=lse_mode,
        min_iterations=0,
        max_iterations=100)
    
    return(ot_solve.primal_cost)

def W1_grad(x, y, eps, lse_mode = False):
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


def W2_norm(x, y, eps, lse_mode = False):
    x,a = x[0], x[1]
    y,b = y[0], y[1]
    
        
    ot_solve = linear.solve(
        ott.geometry.pointcloud.PointCloud(x, y, cost_fn=ott.geometry.costs.Euclidean(), epsilon = eps),
        a = a,
        b = b,
        lse_mode=lse_mode,
        min_iterations=0,
        max_iterations=100)

    return(ot_solve.primal_cost)

def W2_norm_grad(x, y, eps, lse_mode = False):
    x,a = x[0], x[1]
    y,b = y[0], y[1]
    
        
    ot_solve = linear.solve(
        ott.geometry.pointcloud.PointCloud(x, y, cost_fn=ott.geometry.costs.Euclidean(), epsilon = eps),
        a = a,
        b = b,
        lse_mode=lse_mode,
        min_iterations=0,
        max_iterations=100)

    return(ot_solve.reg_ot_cost)
    
def S2_norm(x, y, eps, lse_mode = False):
    x,a = x[0], x[1]
    y,b = y[0], y[1]
        
    ot_solve_xy = linear.solve(
        ott.geometry.pointcloud.PointCloud(x, y, cost_fn=ott.geometry.costs.Euclidean(), epsilon = eps),
        a = a,
        b = b,
        lse_mode=lse_mode,
        min_iterations=0,
        max_iterations=100)

    ot_solve_xx = linear.solve(
    ott.geometry.pointcloud.PointCloud(x, x, cost_fn=ott.geometry.costs.Euclidean(), epsilon = eps),
    a = a,
    b = a,
    lse_mode=lse_mode,
    min_iterations=0,
    max_iterations=100)
    
    ot_solve_yy = linear.solve(
    ott.geometry.pointcloud.PointCloud(y, y, cost_fn=ott.geometry.costs.Euclidean(), epsilon = eps),
    a = b,
    b = b,
    lse_mode=lse_mode,
    min_iterations=0,
    max_iterations=100)
    return(ot_solve_xy.reg_ot_cost - 0.5 * ot_solve_xx.reg_ot_cost - 0.5 * ot_solve_yy.reg_ot_cost)

def W2(x, y, eps, lse_mode = False):
    x,a = x[0], x[1]
    y,b = y[0], y[1]
    
        
    ot_solve = linear.solve(
        ott.geometry.pointcloud.PointCloud(x, y, cost_fn=None, epsilon = eps),
        a = a,
        b = b,
        lse_mode=lse_mode,
        min_iterations=0,
        max_iterations=100)

    return(ot_solve.primal_cost)

def W2_grad(x, y, eps, lse_mode = False):
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
    x,a = x[0], x[1]
    y,b = y[0], y[1]
    geom_xx = pointcloud.PointCloud(x=x, y=x)
    geom_yy = pointcloud.PointCloud(x=y, y=y)
    
    ot_solve = quadratic.gromov_wasserstein.solve(geom_xx, geom_yy, 
                                                  a = a, b = b, 
                                                  epsilon=eps, max_iterations=50)

    return(ot_solve.primal_cost)

def GW_grad(x, y, eps, lse_mode = False):
    x,a = x[0], x[1]
    y,b = y[0], y[1]
    geom_xx = pointcloud.PointCloud(x=x, y=x)
    geom_yy = pointcloud.PointCloud(x=y, y=y)
    
    ot_solve = quadratic.gromov_wasserstein.solve(geom_xx, geom_yy, 
                                                  a = a, b = b, 
                                                  epsilon=eps, max_iterations=50)

    return(ot_solve.reg_gw_cost)


def GS(x, y, eps, lse_mode = False):
    x,a = x[0], x[1]
    y,b = y[0], y[1]
    
    geom_xx = pointcloud.PointCloud(x=x, y=x)
    geom_yy = pointcloud.PointCloud(x=y, y=y)
    
    
    ot_solve_xy = quadratic.gromov_wasserstein.solve(geom_xx, geom_yy, 
                                                  a = a, b = b, 
                                                  epsilon=eps, max_iterations=50)

    ot_solve_xx = quadratic.gromov_wasserstein.solve(geom_xx, geom_xx, 
                                                  a = a, b = a, 
                                                  epsilon=eps, max_iterations=50)
    
    ot_solve_yy = quadratic.gromov_wasserstein.solve(geom_yy, geom_yy, 
                                                  a = b, b = b, 
                                                  epsilon=eps, max_iterations=50)
    return(ot_solve_xy.reg_gw_cost - 0.5 * ot_solve_xx.reg_gw_cost - 0.5 * ot_solve_yy.reg_gw_cost)

def Zeros(x, y, eps, lse_mode = False):
    return(jnp.zeros([x[0].shape[0]]))
