"""
Optimization algorithms.
"""
import pydecode.hyper as ph
import numpy as np
from numpy.linalg import norm


def best_constrained_path(graph, potentials, constraints):
    r"""
    Compute the best constrained path for a hypergraph.

    Parameters
    ----------

    graph : :py:class:`Hypergraph`
      The hypergraph.

    potentials : :py:class:`pydecode.hyper.LogViterbiPotentials`
      The log viterbi potentials.

    constraints : :py:class:`pydecode.constraints.Constraints`
      The constraints on the path.

    Returns
    -------
    g : subgradient vector
      The last term.

    """
    _, _, _, extras = \
        subgradient_descent(_subgradient(graph, potentials, constraints.potentials), [0] * constraints.size, polyak)
    return extras[-1]


def _subgradient(graph, weight_potentials, potentials):
    r"""
    Compute a subgradient with respect to potentials.

    Parameters
    ----------

    graph : :py:class:`Hypergraph`
      The hypergraph.

    weight_potentials : :py:class:`pydecode.hyper.LogViterbiPotentials`
      The log viterbi potentials.

    potentials : :py:class:`pydecode.hyper.Potentials`
      The potential.

    Returns
    -------
    fn : A function
      A function for subgradient descent.
    """
    def fn(x):
        mod_weights = ph.pairwise_dot(potentials, x);
        dual_weights = weight_potentials.times(mod_weights)
        path = ph.best_path(graph, dual_weights)
        score = dual_weights.dot(path)
        vec = potentials.dot(path)
        subgrad = np.zeros(len(x))
        for i in vec:
            subgrad[i] = vec[i]
        return score, subgrad, path
    return fn


def subgradient_descent(fn, x0, rate, max_iterations=100):
    r"""
    Runs subgradient descent on the objective function.

    Assume we have a set :math:`{\cal X}`
    and a function :math:`f: {\cal X}  \rightarrow {\cal R}`
    and a subgradient function :math:`g: {\cal X}  \rightarrow {\cal X}`.

    Parameters
    ----------
    fn : function
      Takes an argument :math:`x \in {\cal X}`.

      Returns a tuple :math:`(f(x), g(x)) \in {\cal R} \times {\cal X}`, consisting of the objective score and a subgradient vector.

    x0 : array
      The initial parameter values :math:`x0 \in {\cal X}`.
    rate : function
      A function :math:`\alpha(t): {\cal X} \rightarrow {\cal R}`.

      Implements the rate parameter for subgradient descent.

    max_iterations : int, optional
      The number of iterations to run.

    Returns
    -------
    xs, ys, subgradients : lists
      intermediate values for xs, ys, gs
    """
    subgradients = []
    f_x_s = []
    xs = [x0]
    extras = []
    x_best = x0
    f_x_best = 1e8

    for t in range(max_iterations):
        x = xs[-1]
        f_x, g, extra = fn(x)

        subgradients.append(g)
        f_x_s.append(f_x)
        if f_x < f_x_best:
            f_x_best = f_x
            x_best = x

        x_plus = x - rate(t, f_x, f_x_best, g) * g
        xs.append(x_plus)
        extras.append(extra)

        if norm(g) == 0: break
    return xs, f_x_s, subgradients, extras


def polyak(t, f_x, f_x_best, g):
    r"""
    Implements the Polyak Rule.
    Described in Boyd note.

    Parameters
    ----------
    t : int
      Round of subgradient descent.

    f_x : vector
      The current objective value.

    f_x_best : real
      The lowest objective value seen.

    g : vector
      Current subgradient vector.
    """
    if norm(g) > 0:
        return (f_x - f_x_best + 1.0/(t+1)) / (g * g)
    else:
        return 0.0
