from collections import namedtuple, defaultdict
import itertools
import pydecode.nlp.decoding as decoding
import pydecode.hyper as ph
import random
import numpy as np

class DependencyDecodingProblem(decoding.DecodingProblem):
    def __init__(self, size, order):
        self.size = size
        self.order = order

    def feasible_set(self):
        """
        Enumerate all possible projective trees.

        Returns
        --------
        parses : iterator
           An iterator of possible n-parse trees.
        """
        n = self.size
        for mid in itertools.product(range(n + 1), repeat=n):
            parse = DependencyParse([-1] + list(mid))
            if (not parse.check_projective()) or (not parse.check_spanning()):
                continue
            yield parse

class DependencyParse(object):
    """
    Class representing a dependency parse with
    possible unused modifier words.
    """

    def __init__(self, heads):
        """
        Parameters
        ----------
        head : List
           The head index of each modifier or None for unused modifiers.
           Requires head[0] = -1 for convention.
        """
        self.heads = heads
        assert(self.heads[0] == -1)

    def __eq__(self, other):
        return self.heads == other.heads

    def __cmp__(self, other):
        return cmp(self.heads, other.heads)

    def __repr__(self):
        return str(self.heads)

    def arcs(self, second=False):
        """
        Returns
        -------
        arc : iterator of (m, h) pairs in m order
           Each of the arcs used in the parse.
        """
        for m, h in enumerate(self.heads):
            if m == 0 or h is None: continue
            yield (m, h)

    def siblings(self, m):
        return [m2 for (m2, h) in self.arcs()
                if h == self.heads[m]
                if m != m2]

    def sibling(self, m):
        sibs = self.siblings(m)
        h = self.heads[m]
        if m > h:
            return max([s2 for s2 in sibs if h < s2 < m] + [h])
        if m < h:
            return min([s2 for s2 in sibs if h > s2 > m] + [h])


    def sequence(self):
        """
        Returns
        -------
        sequence : iterator of m indices in m order
           Each of the words used in the sentence,
           by convention starts with 0 and ends with n+1.
        """
        yield 0
        for m, h in self.arcs():
            yield m
        yield len(self.heads)

    def skipped_words(self):
        return len([h for h in self.heads if h is None])

    def check_spanning(self):
        """
        Is the parse tree as valid spanning tree?

        Returns
        --------
        spanning : bool
           True if a valid spanning tree.
        """
        d = {}
        for m, h in self.arcs():
            if m == h:
                return False

            d.setdefault(h, [])
            d[h].append(m)
        stack = [0]
        seen = set()
        while stack:
            cur = stack[0]
            if cur in seen:
                return False
            seen.add(cur)
            stack = d.get(cur,[]) + stack[1:]
        if len(seen) != len(self.heads) - len([1 for p in self.heads if p is None]):
            return False
        return True

    def check_projective(self):
        """
        Is the parse tree projective?

        Returns
        --------
        projective : bool
           True if a projective tree.
        """

        for m, h in self.arcs():
            for m2, h2 in self.arcs():
                if m2 == m: continue
                if m < h:
                    if m < m2 < h < h2 or m < h2 < h < m2 or \
                            m2 < m < h2 < h or  h2 < m < m2 < h:
                        return False
                if h < m:
                    if h < m2 < m < h2 or h < h2 < m < m2 or \
                            m2 < h < h2 < m or  h2 < h < m2 < m:
                        return False
        return True



class DependencyScorer(decoding.Scorer):
    """
    Object for scoring parse structures.
    """

    def __init__(self, problem, arc_scores, second_order=None):
        """
        Parameters
        ----------
        n : int
           Length of the sentence (without root).

        arc_scores : 2D array (n+1 x n+1)
           Scores for each possible arc
           arc_scores[h][m].

        second_order : 3d array (n+2 x n+2 x n+2)
           Scores for each possible modifier bigram
           second_order[h][s][m].
        """
        self.arc_scores = arc_scores
        self._second_order_scores = second_order
        self._problem = problem

    @staticmethod
    def random(dependency_problem):
        n = dependency_problem.size
        order = dependency_problem.order
        first = [[random.random() for j in range(n+1)] for i in range(n+1)]
        second = None
        if order == 2:
            second = [[[random.random() for k in range(n+1)] for j in range(n+1)]
                      for i in range(n+1)]
        return DependencyScorer(dependency_problem, first, second)



    def arc_score(self, head, modifier, sibling=None):
        """
        Returns
        -------
        score : float
           The score of head->modifier
        """
        assert((sibling is None) or (self._second_order_scores is not None))
        if sibling is None:
            return self.arc_scores[head][modifier]
        else:
            return self.arc_scores[head][modifier] + \
                self._second_order_scores[head][sibling][modifier]

    def score(self, parse):
        """
        Score a parse based on arc score.

        Parameters
        ----------
        parse : Parse
            The parse to score.

        Returns
        -------
        score : float
           The score of the parse structure.
        """
        parse_score = 0.0
        if self._second_order_scores is None:
            parse_score = \
                sum((self.arc_score(h, m)
                     for m, h in parse.arcs()))
        else:
            parse_score = \
                sum((self.arc_score(h, m, parse.sibling(m))
                     for m, h in parse.arcs()))

        return parse_score

# Globals
kShapes = 3
Tri = 0
Trap = 1
Box = 2

kDir = 2
Right = 0
Left = 1

def NodeType(type, dir, s, t) :
    return (type, dir, s, t)
def node_type(nodetype): return nodetype[0]
def node_span(nodetype): return nodetype[2:]
def node_dir(nodetype): return nodetype[1]


def Arc(head, mod):
    return (head, mod)

def Arc2(head, mod, sibling):
    return (head, mod, sibling)

def arc_head(arc):
    return arc[0]
def arc_mod(arc):
    return arc[1]
def arc_sib(arc):
    if len(arc) > 2:
        return arc[2]
    else:
        return None

class DependencyDecoder(decoding.HypergraphDecoder):
    def _to_arc(self, hasher, edge):

        label = edge.head_label

        if label is None: return None
        typ, d, s, t = label.unpack()
        if typ != Trap:
            return None
        if d == Left: s, t = t, s
        return (s, t)

    def path_to_instance(self, problem, path):
        n = problem.size + 1
        print "path"
        #hasher = ph.SizedTupleHasher([3, 2, n, n])
        # hasher = ph.QuartetHasher(3, 2, n, n])

        labels = [self._to_arc(None, edge) for edge in path]
        labels = [l for l in labels if l is not None]
        labels.sort(key = arc_mod)
        heads = ([-1] + [arc_head(arc) for arc in labels])
        return DependencyParse(heads)

    def potentials(self, hypergraph, scorer, problem):
        n = problem.size + 1
        print len(hypergraph.nodes)
        print len(hypergraph.edges)

        scores = np.zeros([len(hypergraph.edges)])
        for edge_num, label in hypergraph.head_labels():
            typ, d, s, t = label.unpack()
            if typ != Trap: continue
            if d == Left: s, t = t, s
            scores[edge_num] = scorer.arc_scores[s][t]
        # def score(edge):
        #     arc = self._to_arc(hasher, edge)
        #     if arc == None: return 0.0
        #     # label = hasher.unhash(edge.head.label)
        #     # s, t = node_span(label)
        #     # d = node_dir(label)
        #     # if d == Left: s, t = t, s
        #     return scorer.arc_score(arc[0], arc[1])
        return ph.LogViterbiPotentials(hypergraph).from_array(scores)
                # score(edge) for edge in hypergraph.edges])

class FirstOrderDecoder(DependencyDecoder):
    def potentials(self, hypergraph, scorer, problem):
        return ph.LogViterbiPotentials(hypergraph).from_array(self.data)

    """
    First-order dependency parsing.
    """
    def dynamic_program(self, c, problem, scorer):
        """
        Parameters
        -----------
        sentence_length : int
          The length of the sentence.

        Returns
        -------
        chart :
          The finished chart.
        """
        n = problem.size + 1
        q = ph.Quartet
        num_edges = 4 * n ** 3
        hasher = ph.QuartetHasher(q(kShapes, kDir, n, n))
        c.set_expected_size(hasher.max_size(), num_edges, max_arity=2)
        c.set_hasher(hasher)
        self.data = np.zeros([num_edges])
        c.set_data(self.data)

        # Add terminal nodes.
        for s in range(n):
            for d in [Right, Left]:
                for sh in [Tri]:
                    c.init(q(sh, d, s, s))

        for k in range(1, n):
            for s in range(n):
                t = k + s
                if t >= n: break

                # First create incomplete items.
                if s != 0:
                    arc_score = scorer.arc_score(t, s)
                    c.set(q(Trap, Left, s, t),
                          [((q(Tri, Right, s, r),
                             q(Tri, Left, r+1, t)), arc_score)
                           for r in xrange(s, t)])

                arc_score = scorer.arc_score(s, t)
                c.set(q(Trap, Right, s, t),
                      [((q(Tri, Right, s, r),
                         q(Tri, Left, r+1, t)), arc_score)
                       for r in xrange(s, t)])

                if s != 0:
                    c.set(q(Tri, Left, s, t),
                          [((q(Tri, Left, s, r),
                             q(Trap, Left, r, t)), 0.0)
                           for r in xrange(s, t)])

                c.set(q(Tri, Right, s, t),
                      [((q(Trap, Right, s, r),
                         q(Tri, Right, r, t)), 0.0)
                       for r in xrange(s + 1, t + 1)])


class SecondOrderDecoder(DependencyDecoder):
    def dynamic_program(self, c, problem):
        n = problem.size + 1
        # hasher = ph.SizedTupleHasher([3, 2, n, n])
        c.set_hasher(hasher)

        # Initialize the chart.
        for s in range(n):
             for d in [Right, Left]:
                 for sh in [Tri]:
                     c.init(NodeType(sh, d, s, s))

        for k in range(1, n):
            for s in range(n):
                t = k + s
                if t >= n: break
                #span = (s, t)

                if s != 0:
                    c.set(NodeType(Box, Left, s, t),
                          [((c[key1], c[key2]), None)
                           for r  in range(s, t)
                           for key1 in [(Tri, Right, s, r)]
                           if key1 in c
                           for key2 in [(Tri, Left, r+1, t)]
                           if key2 in c])

                if s != 0:
                    c.set(NodeType(Trap, Left, s, t),
                          [((c[key1], c[key2]), Arc(t, s, t))
                           for key1 in [(Tri, Right, s, t-1)]
                           if key1 in c
                           for key2 in [(Tri, Left, t, t)]
                           if key2 in c] +
                          [((c[key1], c[key2]),  Arc(t, s, r))
                           for r in range(s+1, t)
                           for key1 in [(Box, Left, s, r)]
                           if key1 in c
                           for key2 in [(Trap, Left, r, t)]
                           if key2 in c])

                c.set(NodeType(Trap, Right, s, t),
                      [((c[key1], c[key2]),  Arc(s, t, s))
                       for key1 in [(Tri, Right, s, s)]
                       if key1 in c
                       for key2 in [(Tri, Left, s+1, t)]
                       if key2 in c] +
                      [((c[key1], c[key2]), Arc(s, t, r))
                       for r  in range(s + 1, t)
                       for key1 in [(Trap, Right, s, r)]
                       if key1 in c
                       for key2 in [(Box, Left, r, t)]
                       if key2 in c])

                if s != 0:
                    c.set(NodeType(Tri, Left, s, t),
                          [((c[key1], c[key2]), None)
                           for r  in range(s, t)
                           for key1 in [(Tri, Left, s, r)]
                           if key1 in c
                           for key2 in [(Trap, Left, r, t)]
                           if key2 in c])

                c.set(NodeType(Tri, Right, s, t),
                      [((c[key1], c[key2]), None)
                      for r in range(s + 1, t + 1)
                      for key1 in [(Trap, Right, s, r)]
                      if key1 in c
                      for key2 in [(Tri, Right, r, t)]
                      if key2 in c])
