from collections import namedtuple
import itertools
import pydecode.nlp.decoding as decoding
import pydecode.constraints as constraints
import pydecode as ph
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


class PermutationScorer(object):
    @staticmethod
    def random(dependency_problem):
        n = dependency_problem.size
        return numpy.random.random([n, n])

    @staticmethod
    def score(scores, perm):
        return sum((scores[i, j]
                 for i, j in perm.transition()))

def make_lattice(width, height, transitions):
    w, h = width, height

    blank = np.array([], dtype=np.int64)

    coder = np.arange(w * h, dtype=np.int64)\
        .reshape([w+2, h])
    out = np.arange(w * h * h, dtype=np.int64)\
        .reshape([w, h, h])

    c = ph.ChartBuilder(coder.size,
                        unstrict=True,
                        output_size=out.size)

    c.init(coder[0, 0])
    for i in range(1, w + 1):
        for j in range(h):
            c.set2(coder[i, j],
                   coder[i-1, transitions[j]],
                   blank,
                   out[i-1, j, transitions[j]])
    c.set(coder[w+1, 0],
          coder[w, :h],
          blank,
          blank)
    return c


class PermutationDecoder(decoding.ConstrainedHypergraphDecoder):
    def output_to_instance(self, problem, items):
        w, h = problem.size-1, problem.size
        trans = numpy.unravel_index(items.nonzero()[0], (w, h, h))
        trans = zip(*trans)
        perms = [-1] * (problem.size - 1)
        for i, j, _ in trans:
            perms[i] = j
        return Permutation(perms)

    def constraints(self, hypergraph, problem):
        cons = constraints.Constraints(hypergraph,
                                       [(i, -1) for i in range(problem.size)])
        def make_constraint(edge):
            if edge.head.label.i == 0 or edge.head.label.i > problem.size:
                return []
            return [(edge.head.label.j, 1)]

        cons.from_vector([make_constraint(edge)
                          for edge in hypergraph.edges])
        return cons

    def hypergraph(self, problem):
        return make_lattice(problem.size-1, problem.size,
                            np.repeat(np.arange(problem.size), problem.size))

    def special_decode(self, method, problem, hypergraph, scores, constraints,
                       scorer):
        if method == "CUBE":
            groups = [node.label.i for node in hypergraph.nodes]
            ins = ph.inside(hypergraph, scores)
            out = ph.outside(hypergraph, scores, ins)

            beam_chart = ph.beam_search_BinaryVector(
                hypergraph, scores, constraints.to_binary_potentials(),
                out, -10000, groups, [1000] * len(groups), cube_pruning=True)
            return beam_chart.path(0)

        elif method == "BEAM":
            groups = [node.label.i for node in hypergraph.nodes]
            ins = ph.inside(hypergraph, scores)
            out = ph.outside(hypergraph, scores, ins)

            beam_chart = ph.beam_search_BinaryVector(
                hypergraph, scores, constraints.to_binary_potentials(),
                out, -10000, groups, [1000] * len(groups))
            return beam_chart.path(0)
        elif method == "MULTIDFA":
            old = hypergraph
            old_hmap = None

            for j in range(problem.size):
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
                #new_scores = old_scores.up_project(old, hmap)
                if old_hmap is not None:
                    old_hmap = old_hmap.compose(hmap)
                else:
                    old_hmap = hmap
                # old_scores = new_scores
            new_scores = scores.up_project(old, old_hmap)
            #new_scores = self.potentials(old, scorer)
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
            new_scores = scores.up_project(old, hmap)
            return ph.best_path(old, new_scores)
