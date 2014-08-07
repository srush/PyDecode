import numpy as np
import pydecode
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

def decode_exhaustive(decoding_problem, scores, coder):
    best_score = MIN
    best = None
    for y in decoding_problem.feasible_set():
        output = coder.transform(y)
        indices = np.ravel_multi_index(output.T, coder.shape_)
        score = np.sum(scores.ravel()[indices])
        if score > best_score:
            best_score = score
            best = y
    return best

class HypergraphDecoder(object):
    def output_coder(self, problem):
        raise NotImplementedError()

    def dynamic_program(self, problem):
        raise NotImplementedError()

    def decode(self, decoding_problem, scores):
        """
        """
        dp = self.dynamic_program(decoding_problem)
        return self.output_coder(decoding_problem).\
            inverse_transform(pydecode.argmax(dp, scores))

# TODO: update this

# class ConstrainedHypergraphDecoder(Decoder):
#     def __init__(self, method="ILP"):
#         self.method = method

#     def hypergraph(self, problem):
#         raise NotImplementedError()

#     def special_decode(self):
#         raise NotImplementedError()

#     def decode(self, decoding_problem, scorer):
#         hypergraph = self.hypergraph(decoding_problem)
#         scores = self.potentials(hypergraph, scorer, decoding_problem)
#         constraints = self.constraints(hypergraph, decoding_problem)


#         if self.method == "ILP":
#             hyperlp = lp.HypergraphLP.make_lp(hypergraph,
#                                               scores,
#                                               integral=True)
#             hyperlp.add_constraints(constraints)
#             hyperlp.solve(pulp.solvers.GLPK(mip=1, msg=0))
#             path = hyperlp.path
#         else:
#             path = self.special_decode(self.method, decoding_problem,
#                                        hypergraph, scores, constraints, scorer)

#         return self.path_to_instance(decoding_problem, path)
