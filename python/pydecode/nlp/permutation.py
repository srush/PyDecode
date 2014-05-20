from collections import namedtuple
import itertools
import pydecode.nlp.decoding as decoding
import pydecode.constraints as constraints
import pydecode.hyper as ph
import random

class PermutationProblem(decoding.DecodingProblem):
    def __init__(self, size):
        self.size = size

    def feasible_set(self):
        for mid in itertools.permutations(range(1, self.size)):
            perm = Permutation(list(mid))
            assert perm.check_valid()
            yield perm

class Permutation(object):
    def __init__(self, perm):
        self.perm = perm

    def __eq__(self, other):
        return self.perm == other.perm

    def __cmp__(self, other):
        return cmp(self.perm, other.perm)

    def __repr__(self):
        return str(self.perm)

    def transition(self):
        yield (0, self.perm[0])
        for r in itertools.izip(self.perm, self.perm[1:]):
            yield r
        yield (self.perm[-1], 0)


    def check_valid(self):
        d = set()
        for i in self.perm:
            if i in d: return False
            d.add(i)
        return True


class PermutationScorer(decoding.Scorer):
    """
    Object for scoring permutation structures.
    """

    def __init__(self, problem, bigram_scores):
        self._bigram_score = bigram_scores
        self._problem = problem

    @staticmethod
    def random(dependency_problem):
        n = dependency_problem.size
        first = [[random.random() for j in range(n)] for i in range(n)]
        return PermutationScorer(dependency_problem, first)

    def trans_score(self, i, j):
        return self._bigram_score[i][j]

    def score(self, perm):
        return \
            sum((self.trans_score(i, j)
                 for i, j in perm.transition()))


class PermutationDecoder(decoding.ConstrainedHypergraphDecoder):
    def path_to_instance(self, problem, path):
        perms = [-1] * (problem.size - 1)
        for edge in path:
            if edge.head.label.i == 0 or edge.head.label.i > problem.size-1: continue
            perms[edge.head.label.i - 1] = edge.head.label.j
        return Permutation(perms)

    def potentials(self, hypergraph, scorer):
        def score(edge):
            return scorer.trans_score(edge.tail[0].label.j, edge.head.label.j)

        return ph.LogViterbiPotentials(hypergraph).from_vector([
                score(edge) for edge in hypergraph.edges])

    def constraints(self, hypergraph, problem):
        cons = constraints.Constraints(hypergraph,
                                       [(i, -1) for i in range(problem.size)])
        def make_constraint(edge):
            if edge.head.label.i == 0 or edge.head.label.i > problem.size: return []
            return [(edge.head.label.j, 1)]

        cons.from_vector([make_constraint(edge) for edge in hypergraph.edges])
        return cons

    def hypergraph(self, problem):
        return ph.make_lattice(problem.size-1, problem.size,
                               [range(problem.size) for _ in range(problem.size)])

    def special_decode(self, method, problem, hypergraph, scores, constraints,
                       scorer):
        if method == "BEAM":
            groups = [node.label.i for node in hypergraph.nodes]
            ins = ph.inside(hypergraph, scores)
            out = ph.outside(hypergraph, scores, ins)

            beam_chart = ph.beam_search_BinaryVector(
                hypergraph, scores, constraints.to_binary_potentials(),
                out, -10000, groups, [1000] * len(groups))
            return beam_chart.path(0)
        elif method == "MULTIDFA":
            old = hypergraph
            for j in range(problem.size):
                print j
                states = 2
                symbols = 2
                dfa = ph.DFA(states, symbols, [{0:0, 1:1} , {0:1}], [1])
                vec = [(1 if (edge.head.label.j == j) else 0)
                       for edge in old.edges]
                counts = ph.CountingPotentials(old).from_vector(vec)
                hmap = ph.extend_hypergraph_by_dfa(old, counts, dfa)
                old = hmap.domain_hypergraph
                old.labeling = ph.Labeling(old, [hmap[node].label
                                                 for node in old.nodes],
                                           None)
            new_scores = self.potentials(old, scorer)
            return ph.best_path(old, new_scores)

        elif method == "BIGDFA":
            old = hypergraph
            states = 2**problem.size
            symbols = problem.size + 1
            final_state = 0
            for i in range(problem.size):
                final_state |= 2**i

            transitions = \
                [{j : i | 2**j for j in range(symbols) if i & 2**j == 0}
                 for i in range(states)]
            dfa = ph.DFA(states, symbols,
                         transitions,
                         [final_state])
            vec = [edge.head.label.j for edge in old.edges]
            counts = ph.CountingPotentials(old).from_vector(vec)
            hmap = ph.extend_hypergraph_by_dfa(old, counts, dfa)
            old = hmap.domain_hypergraph
            old.labeling = ph.Labeling(old, [hmap[node].label
                                             for node in old.nodes],
                                       None)
            new_scores = self.potentials(old, scorer)
            return ph.best_path(old, new_scores)
