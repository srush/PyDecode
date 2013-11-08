"""
Optimization algorithms.
"""
import pydecode.hyper as ph
import numpy as np
from numpy.linalg import norm

def subgradient(graph, potentials, constraints):
    def fn(x):
        mod_potentials = ph.pairwise_dot(constraints, x);
        dual_potentials = potentials.times(mod_potentials)
        path = ph.best_path(graph, dual_potentials)
        score = dual_potentials.dot(path)
        vec = constraints.dot(path)
        subgrad = np.zeros(len(x))
        for i in vec:
            subgrad[i] = vec[i]
        return score, subgrad, path
    return fn

def best_constrained_path(graph, potentials, constraints):
    _, _, _, extras = subgradient_descent(subgradient(graph, potentials, constraints.potentials), [0] * constraints.size, polyak)
    return extras[-1]

def subgradient_descent(fn, x0, rate, max_iterations=100):
    """
    Runs subgradient descent on the objection function

    Parameters
    ----------
    fn : function
      Computes the objective function and a subgradient.
    x0 : array
      The initial parameter values.
    rate : function
      The rate parameter alpha_t.
    max_iterations : int, optional
      The number of iterations to run.

    Returns
    -------
    xs, ys, subgradients : lists
      intermediate values for xs, ys, gs
    """
    subgradients = []
    ys = []
    xs = [x0]
    extras = []
    x_best = x0
    y_best = 1e8

    for t in range(max_iterations):
        x = xs[-1]
        y, g, extra = fn(x)

        subgradients.append(g)
        ys.append(y)
        if y < y_best:
            y_best = y
            x_best = x

        x_plus = x - rate(t, y, y_best, g) * g
        xs.append(x_plus)
        extras.append(extra)

        if norm(g) == 0: break
    return xs, ys, subgradients, extras


def polyak(t, f_x, f_x_best, g):
    if norm(g) > 0:
        return (f_x - f_x_best + 1.0/(t+1)) / (g * g)
    else:
        return 0.0
