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

class SubgradientRoundResult:
    def __init__(self, dual, subgrad, extra):
        self.dual = dual
        self.subgrad = subgrad
        self.extra = extra

class SubgradientRoundStatus:
    def __init__(self, dual, primal, x, x_diff,
                 updated, result, round):
        self.dual = dual
        self.primal = primal
        self.x = x
        self.x_diff = x_diff
        self.updated = updated
        self.result = result
        self.round = round

class SubgradientHistory:
    def __init__(self):
        self.best_primal = -1e8
        self.best_dual = 1e8
        self.history = []

    def status(self):
        return self.history[-1]

    def update(self, status):
        self.history.append(status)
        if status.primal is not None and \
                status.primal > self.best_primal:
            self.best_primal = status.primal
        if status.dual is not None and \
                status.dual < self.best_dual:
            self.best_dual = status.dual

    def show(self):
        status = self.status()
        print "%d %4.3f %4.3f %4.3f %4.3f %4.3f" \
            % (status.round, status.dual, status.primal,
               self.best_primal, self.best_dual,
               abs(self.best_primal - self.best_dual))

def _subgradient(graph, weight_potentials, potentials, best_path_fn=ph.best_path):
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
    dual_weights = weight_potentials.clone()
    chart = ph.LogViterbiChart(graph)
    node_updates = ph.NodeUpdates(graph, potentials)
    dynamic_viterbi = ph.LogViterbiDynamicViterbi(graph)
    dynamic = False
    def fn(status):
        if status.x_diff is None:
            ph.pairwise_dot(potentials, status.x, dual_weights)
            if dynamic:
                dynamic_viterbi.initialize(dual_weights)
        else:
            ph.pairwise_dot(potentials, status.x_diff, dual_weights)

            if dynamic:
                updates = set([i for i, x in enumerate(status.x_diff)
                               if x != 0.0])
                up = node_updates.update(updates)
                print "num nodes updated", len(up), len(graph.nodes)
                dynamic_viterbi.update(dual_weights, up)
        if dynamic:
            path = dynamic_viterbi.path
        else:
            path = best_path_fn(graph, dual_weights, chart)

        #score = dual_weights.dot(path)
        score = dual_weights.dot(path)
        #print "score1: %f score2: %f"%(score, score2)
        #assert score == score2
        vec = potentials.dot(path)
        subgrad = np.zeros(len(status.x))
        for i, j in vec:
            subgrad[i] = j
        return SubgradientRoundResult(
            dual=score, subgrad=subgrad, extra=path)
    return fn




def subgradient_descent(fn, x0, rate, max_iterations=100,
                        primal_fn=lambda r, extra:None):
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
    history = SubgradientHistory()
    status = SubgradientRoundStatus(x=x0,
                                    x_diff=None,
                                    dual=1e8,
                                    updated=None,
                                    primal=-1e8,
                                    round=0,
                                    result= None)
    history.update(status)
    for t in range(max_iterations):

        result = fn(status)

        x_diff = -rate(history) * result.subgrad
        updates = x_diff != 0
        x_plus = status.x + x_diff

        primal = primal_fn(status, result)
        status = \
            SubgradientRoundStatus(x=x_plus,
                                   x_diff=x_diff,
                                   dual=result.dual,
                                   round=t,
                                   updated=updates,
                                   primal=primal,
                                   result=result)
        history.update(status)
        history.show()
        if norm(result.subgrad) == 0: break
    return history


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
