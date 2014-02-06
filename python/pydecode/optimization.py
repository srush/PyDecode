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
    def __init__(self, dual, subgrad, extra, prune, constraints):
        self.dual = dual
        self.subgrad = subgrad
        self.extra = extra
        self.prune = prune
        self.constraints= constraints

class SubgradientRoundStatus:
    def __init__(self, dual, primal, x, x_diff,
                 updated, result, round, rate):
        self.dual = dual
        self.primal = primal
        self.x = x
        self.x_diff = x_diff
        self.updated = updated
        self.result = result
        self.round = round
        self.rate = rate

class SubgradientHistory:
    def __init__(self):
        self.best_primal = -1e8
        self.best_dual = 1e8
        self.history = []
        self.last_prune = 45
        self.success = False

    def gap(self):
        return abs(self.best_primal - self.best_dual)


    def status(self):
        return self.history[-1]

    def update(self, status):
        if status.result is not None and status.result.prune is not None:
            self.last_prune = status.result.prune
        self.history.append(status)
        if status.primal is not None and \
                status.primal > self.best_primal:
            self.best_primal = status.primal
        if status.dual is not None and \
                status.dual < self.best_dual:
            self.best_dual = status.dual
        assert self.best_primal < self.best_dual, "%f %f"%(self.best_primal, self.best_dual)

    def show(self):
        status = self.status()
        print "SUBGRAD %d %4.3f %4.3f %4.3f %4.3f %4.3f  %4.3f" \
            % (status.round, status.dual, status.primal,
               self.best_primal, self.best_dual, self.gap(), status.rate)

class SubgradientGenerator:
    def __init__(self, graph, weight_potentials, potentials):
        self.dual_weights = weight_potentials.clone()
        self.chart = ph.LogViterbiChart(graph)
        self.current_graph = graph
        self.pruning_function = None
        self.potentials = potentials
        self.weight_potentials = weight_potentials

    def __call__(self, history):
        status = history.status()
        if status.x_diff is None:
            ph.pairwise_dot(self.potentials, status.x, self.dual_weights)
        else:
            ph.pairwise_dot(self.potentials, status.x_diff, self.dual_weights)

        path = ph.best_path(self.current_graph,
                            self.dual_weights)

        score = self.dual_weights.dot(path)
        vec = self.potentials.dot(path)
        subgrad = np.zeros(len(status.x))
        prune = None

        # Call prune function
        #if self.use_prune and history.gap() < history.last_prune - 5:
        if self.pruning_function is not None:
            #print "pruning "
            pruning = \
                self.pruning_function(self.current_graph,
                                      self.dual_weights,
                                      self.potentials,
                                      history)

            # pruning = ph.prune_hypergraph(self.current_graph,
            #                               self.dual_weights,
            #                               history.best_primal)

            if pruning is not None:
                graph, dual_weights, potentials = pruning
                self.dual_weights = dual_weights
                self.potentials = potentials
                print "OLD SIZE", len(self.current_graph.nodes)
                self.current_graph = graph
                print "NEW SIZE", len(self.current_graph.nodes)
                prune = history.gap()

        for i, j in vec:
            subgrad[i] = j

        return SubgradientRoundResult(
            dual=score, subgrad=subgrad,
            extra=path, prune=prune,
            constraints=vec)


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
    return SubgradientGenerator(graph, weight_potentials, potentials)




def subgradient_descent(fn, x0, rate, max_iterations=100,
                        primal_fn=lambda r,extra:None, primal_start=-1e8, ):
    r"""
    Runs subgradient descent on the objective function.

    Assume we have a set :math:`{\cal X}`
    and a function :math:`f: {\cal X}  \rightarrow {\cal R}`
    and a subgradient function :math:`g: {\cal X}  \rightarrow {\cal X}`.

    Parameters
    ----------
    fn : function
      Takes an argument :math:`x \in {\cal X}`.

      Returns a tuple :math:`(f(x), g(x)) \in {\cal R} \times {\cal
      X}`, consisting of the objective score and a subgradient vector.

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
                                    primal=primal_start,
                                    round=0,
                                    result=None,
                                    rate=0)
    history.update(status)
    for t in range(max_iterations):

        result = fn(history)

        round_rate = rate(history, result)
        x_diff = -round_rate * result.subgrad
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
                                   result=result,
                                   rate=round_rate)

        history.update(status)
        history.show()
        if norm(result.subgrad) == 0:
            history.success = True
            break
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
