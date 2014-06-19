
import pydecode as ph
import pydecode.lp as lp
import pulp

MAX = float('inf')
MIN = float('-inf')

class DecodingProblem(object):
    """
    Representation of a decoding problem of the form :math:`\max_y f(y)`.
    """
    def feasble_set(self):
        """
        The set of feasible solutions for the problem.

        """
        pass

    @staticmethod
    def random():
        raise NotImplementedErrro()

class Scorer(object):
    def score(self, y):
        """
        The score of an instance of the problem.
        """
        pass

    @staticmethod
    def random(decoding_problem):
        raise NotImplementedError()

class Decoder(object):
    def decode(self, decoding_problem, scorer):
        """
        Returns the highest scoring structure.

        Parameters
        ----------

        Returns
        ---------
        """
        raise NotImplementedError()

class ExhaustiveDecoder(Decoder):
    def decode(self, decoding_problem, scorer):
        score = MIN
        best = None
        for y in decoding_problem.feasible_set():
            if scorer.score(y) > score:
                score = scorer.score(y)
                best = y
        return best

class BaseHypergraphDecoder(Decoder):

    def potentials(self, hypergraph, scorer):
        raise NotImplementedError()

    def path_to_instance(self, problem, path):
        raise NotImplementedError()

class HypergraphDecoder(Decoder):
    def dynamic_program(self, chart, problem):
        raise NotImplementedError()

    def decode(self, decoding_problem, scorer):
        # c = chart.ChartBuilder(lambda a:a, chart.HypergraphSemiRing,
        #                        build_hypergraph = True)
        hypergraph = self.dynamic_program(None, decoding_problem, scorer)
        scores = self.potentials(hypergraph, scorer, decoding_problem)
        path = ph.best_path(hypergraph, scores)
        return self.path_to_instance(decoding_problem, path)


class ConstrainedHypergraphDecoder(Decoder):
    def __init__(self, method="ILP"):
        self.method = method

    def hypergraph(self, problem):
        raise NotImplementedError()

    def special_decode(self):
        raise NotImplementedError()

    def decode(self, decoding_problem, scorer):
        hypergraph = self.hypergraph(decoding_problem)
        scores = self.potentials(hypergraph, scorer, decoding_problem)
        constraints = self.constraints(hypergraph, decoding_problem)


        if self.method == "ILP":
            hyperlp = lp.HypergraphLP.make_lp(hypergraph,
                                              scores,
                                              integral=True)
            hyperlp.add_constraints(constraints)
            hyperlp.solve(pulp.solvers.GLPK(mip=1, msg=0))
            path = hyperlp.path
        else:
            path = self.special_decode(self.method, decoding_problem,
                                       hypergraph, scores, constraints, scorer)

        return self.path_to_instance(decoding_problem, path)
