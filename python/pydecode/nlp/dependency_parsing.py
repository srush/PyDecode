from collections import namedtuple
import itertools
import pydecode.nlp.decoding as decoding
import pydecode.hyper as ph
import random

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
        self._arc_scores = arc_scores
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
            return self._arc_scores[head][modifier]
        else:
            return self._arc_scores[head][modifier] + \
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
Tri = 1
Trap = 2
Box = 3

Right = 0
Left = 1

def NodeType(type, dir, span) :
    return (type, dir, span)
def node_type(nodetype): return nodetype[0]
def node_span(nodetype): return nodetype[2]
def node_dir(nodetype): return nodetype[1]


class Arc(namedtuple("Arc", ["head", "mod", "sibling"])):
    def __new__(cls, head, mod, sibling=None):
        return super(Arc, cls).__new__(cls, head, mod, sibling)

class DependencyDecoder(decoding.HypergraphDecoder):
    def path_to_instance(self, problem, path):
        labels =  [edge.label for edge in path if edge.label is not None]
        labels.sort(key = lambda arc: arc.mod)
        heads = ([-1] + [arc.head for arc in labels])
        return DependencyParse(heads)

    def potentials(self, hypergraph, scorer):
        def score(label):
            if label is None: return 0.0
            return scorer.arc_score(label.head, label.mod, label.sibling)
        return ph.LogViterbiPotentials(hypergraph).from_vector([
                score(edge.label) for edge in hypergraph.edges])


class FirstOrderDecoder(DependencyDecoder):
    """
    First-order dependency parsing.
    """
    def dynamic_program(self, c, problem):
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

        # Add terminal nodes.
        [c.init(NodeType(sh, d, (s, s)))
         for s in range(n)
         for d in [Right, Left]
         for sh in [Trap, Tri]]

        for k in range(1, n):
            for s in range(n):
                t = k + s
                if t >= n: break
                span = (s, t)

                # First create incomplete items.
                if s != 0:
                    c[NodeType(Trap, Left, span)] = \
                        c.sum([c[key1] * c[key2] * c.sr(Arc(t, s))
                               for r in range(s, t)
                               for key1 in [(Tri, Right, (s, r))]
                               if key1 in c
                               for key2 in [(Tri, Left, (r+1, t))]
                               if key2 in c
                               ])


                c[NodeType(Trap, Right, span)] = \
                    c.sum([ c[key1] * c[key2] * c.sr(Arc(s, t))
                            for r in range(s, t)
                            for key1 in [(Tri, Right, (s, r))]
                            if key1 in c
                            for key2 in [(Tri, Left, (r+1, t))]
                            if key2 in c])

                if s != 0:
                    c[NodeType(Tri, Left, span)]= \
                        c.sum([c[key1] * c[key2]
                               for r in range(s, t)
                               for key1 in [(Tri, Left, (s, r))]
                               if key1 in c
                               for key2 in [(Trap, Left, (r, t))]
                               if key2 in c])

                c[NodeType(Tri, Right, span)] = \
                    c.sum([c[key1] * c[key2]
                           for r in range(s + 1, t + 1)
                           for key1 in [(Trap, Right, (s, r))]
                           if key1 in c
                           for key2 in [(Tri, Right, (r, t))]
                           if key2 in c])


class SecondOrderDecoder(DependencyDecoder):
    def dynamic_program(self, c, problem):
        n = problem.size + 1

        # Initialize the chart.
        for s in range(n):
             for d in [Right, Left]:
                 for sh in [Tri]:
                     c.init(NodeType(sh, d, (s, s)))

        for k in range(1, n):
            for s in range(n):
                t = k + s
                if t >= n: break
                span = (s, t)

                if s != 0:
                    c[NodeType(Box, Left, span)] = \
                        c.sum([c[key1] * c[key2]
                               for r  in range(s, t)
                               for key1 in [(Tri, Right, (s, r))]
                               if key1 in c
                               for key2 in [(Tri, Left, (r+1, t))]
                               if key2 in c])

                if s != 0:
                    c[NodeType(Trap, Left, span)] = \
                        c.sum([c[key1] * c[key2] * c.sr(Arc(t, s, t))
                               for key1 in [(Tri, Right, (s, t-1))]
                               if key1 in c
                               for key2 in [(Tri, Left, (t, t))]
                               if key2 in c] +
                              [c[key1] * c[key2] * c.sr(Arc(t, s, r))
                               for r in range(s+1, t)
                               for key1 in [(Box, Left, (s, r))]
                               if key1 in c
                               for key2 in [(Trap, Left, (r, t))]
                               if key2 in c])

                c[NodeType(Trap, Right, span)] = \
                    c.sum([c[key1] * c[key2] * c.sr(Arc(s, t, s))
                           for key1 in [(Tri, Right, (s, s))]
                           if key1 in c
                           for key2 in [(Tri, Left, (s+1, t))]
                           if key2 in c] +
                          [c[key1] * c[key2] * c.sr(Arc(s, t, r))
                           for r  in range(s + 1, t)
                           for key1 in [(Trap, Right, (s, r))]
                           if key1 in c
                           for key2 in [(Box, Left, (r, t))]
                           if key2 in c])

                if s != 0:
                    c[NodeType(Tri, Left, span)] = \
                        c.sum([c[key1] * c[key2]
                               for r  in range(s, t)
                               for key1 in [(Tri, Left, (s, r))]
                               if key1 in c
                               for key2 in [(Trap, Left, (r, t))]
                               if key2 in c])

                c[NodeType(Tri, Right, span)] = \
                    c.sum([c[key1] * c[key2]
                           for r in range(s + 1, t + 1)
                           for key1 in [(Trap, Right, (s, r))]
                           if key1 in c
                           for key2 in [(Tri, Right, (r, t))]
                           if key2 in c])
