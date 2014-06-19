from collections import namedtuple, defaultdict
import itertools
import pydecode.nlp.decoding as decoding
import pydecode as ph
import random
import numpy as np
import scipy.sparse

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
        typ, d, s, t = label
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
        n = problem.size + 1
        #index_set = ph.IndexedEncoder((n, n))
        # data = []
        # indices = []
        # ind = [0]
        # for i, ls in enumerate(self.data):
        #     data += [1] * len(ls)
        #     indices += ls
        #     ind.append(len(data))
        # # print data, indices, ind
        # arcs = scipy.sparse.csc_matrix(
        #     (data, indices, ind),
        #     shape=(hasher.max_size(), len(hypergraph.edges)),
        #     dtype=np.uint8)

        scores = np.zeros([n, n])
        for i in range(n):
            for j in range(n):
                scores[i, j] = scorer.arc_score(i, j)
        scores = scores.reshape([1,n*n])
        # print arcs.shape
        # print scores.shape
        # self.data = scores * self.data

        print scores.shape
        print self.data.shape
        print self.data.shape
        print scores.shape
        return scores * self.data

    def dynamic_program(self, p, problem, _):
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
        num_edges = 4 * n ** 3
        item_encoder = ph.IndexedEncoder([kShapes, kDir, n, n])
        coder = np.arange(item_encoder.max_size, dtype=np.int64).reshape([kShapes, kDir, n, n])

        output_encoder = ph.IndexedEncoder((n, n))
        out = np.arange(output_encoder.max_size, dtype=np.int64).reshape([n, n])
        c = ph.ChartBuilder(
            coder, item_encoder.max_size,
            unstrict=True,
            output_encoder=output_encoder,
            output_size= output_encoder.max_size,
            expected_size=(num_edges, 2))

        # Add terminal nodes.
        for s in range(n):
            for d in [Right, Left]:
                for sh in [Tri]:
                    c.init(coder[sh, d, s, s])
                    #p = coder[sh, d, s, s]

        #arr = np.zeros([n+1, 8], dtype=np.int, order="F")

        for k in range(1, n):
            for s in range(n):
                t = k + s
                if t >= n: break

                # First create incomplete items.
                if s != 0:
                    c.set2(coder[Trap, Left, s, t],
                           coder[Tri, Right, s, s:t],
                           coder[Tri, Left, s+1:t+1, t],
                           np.repeat(out[t, s], t-s))

                    # for i, r in enumerate(range(s, t)):
                        # for j in range(8): arr[i,j] = p[j]

                    #np.array(p)

                # p = []
                # for i, r in enumerate(range(s, t)):
                c.set2(coder[Trap, Right, s, t],
                       coder[Tri, Right, s, s:t],
                       coder[Tri, Left, s+1:t+1, t],
                       np.repeat(out[s, t], t -s)
                       )
                    # for j in range(8): arr[i,j] = p[j]

                # c[Trap, Right, s, t] = np.array(p) #arr[:i+1, :]

                # p = []
                if s != 0:
                    # for i, r in enumerate(range(s, t)):
                    c.set2(coder[Tri, Left, s, t],
                           coder[Tri, Left, s, s:t],
                           coder[Trap, Left, s:t, t],
                           np.repeat(0, t -s))
                        # for j in range(8): arr[i,j] = p[j]
                    #np.array(p)
                    # c[Tri, Left, s, t] = np.array(p)#arr[:i+1, :]

                # p = []
                # for i, r in enumerate(range(s + 1, t + 1)):
                # print np.vstack([coder[Trap, Right, s, s+1:t+1],
                #                  coder[Tri, Right, s+t:t+1, t]])
                c.set2(coder[Tri, Right, s, t],
                       coder[Trap, Right, s, s+1:t+1],
                       coder[Tri, Right, s+1:t+1, t],
                       np.repeat(0, t -s))
                    # for j in range(8): arr[i,j] = p[j]

                # c[Tri, Right, s, t] = np.array(p) #arr[:i+1, :]

        hyper = c.finish(False)
        #self.data = c.output_matrix

        return c

    # """
    # First-order dependency parsing.
    # """
    # def dynamic_program(self, p, problem, _):
    #     """
    #     Parameters
    #     -----------
    #     sentence_length : int
    #       The length of the sentence.

    #     Returns
    #     -------
    #     chart :
    #       The finished chart.
    #     """
    #     n = problem.size + 1
    #     num_edges = 4 * n ** 3
    #     item_encoder = ph.IndexedEncoder((kShapes, kDir, n, n))
    #     coder = np.arange(item_encoder.max_size).reshape([kShapes, kDir, n, n])
    #     output_encoder = ph.IndexedEncoder((n, n))
    #     outcoder = np.arange(output_encoder.max_size).reshape([n, n])
    #     c = ph.ChartBuilder(
    #         coder, item_encoder.max_size,
    #         unstrict=True,
    #         output_encoder=outcoder,
    #         output_size= output_encoder.max_size,
    #         expected_size=(num_edges, 2))

    #     # Add terminal nodes.
    #     for s in range(n):
    #         for d in [Right, Left]:
    #             for sh in [Tri]:
    #                 c[sh, d, s, s] = c.init()

    #     for k in range(1, n):
    #         for s in range(n):
    #             t = k + s
    #             if t >= n: break

    #             # First create incomplete items.
    #             if s != 0:
    #                 c[Trap, Left, s, t] = \
    #                       [c.merge((Tri, Right, s, r), (Tri, Left, r+1, t),
    #                        out=[(t, s)])
    #                        for r in range(s, t)]

    #             c[Trap, Right, s, t] = \
    #                   [c.merge((Tri, Right, s, r), (Tri, Left, r+1, t),
    #                            out=[(s, t)])
    #                    for r in range(s, t)]

    #             if s != 0:
    #                 c[Tri, Left, s, t] = \
    #                     [c.merge((Tri, Left, s, r), (Trap, Left, r, t), out=[])
    #                      for r in range(s, t)]

    #             c[Tri, Right, s, t] = \
    #                   [c.merge((Trap, Right, s, r), (Tri, Right, r, t),
    #                            out=[])
    #                    for r in range(s + 1, t + 1)]

    #     hyper = c.finish(False)
    #     self.data = c.output_matrix

    #     return c
    #     # hyper = c.finish(False)
    #     # #self.data = np.zeros([num_edges])
    #     # # print len(hyper.edges)
    #     # # print self.data
    #     # # print self.data
    #     # #print c.matrix().shape
    #     # #self.data = c.matrix()
    #     # return hyper

    # def dynamic_program(self, p, problem, _):
    #     """
    #     Parameters
    #     -----------
    #     sentence_length : int
    #       The length of the sentence.

    #     Returns
    #     -------
    #     chart :
    #       The finished chart.
    #     """
    #     n = problem.size + 1
    #     num_edges = 4 * n ** 3
    #     item_encoder = ph.IndexedEncoder((kShapes, kDir, n, n))
    #     coder = np.arange(item_encoder.max_size).reshape([kShapes, kDir, n, n])
    #     output_encoder = ph.IndexedEncoder((n, n))
    #     outcoder = np.arange(output_encoder.max_size).reshape([n, n])
    #     c = ph.ChartBuilder(
    #         coder, item_encoder.max_size,
    #         unstrict=True,
    #         output_encoder=outcoder,
    #         output_size= output_encoder.max_size,
    #         expected_size=(num_edges, 2))

    #     # Add terminal nodes.
    #     for s in range(n):
    #         for d in [Right, Left]:
    #             for sh in [Tri]:
    #                 c[sh, d, s, s] = c.init()

    #     for k in range(1, n):
    #         for s in range(n):
    #             t = k + s
    #             if t >= n: break

    #             # First create incomplete items.
    #             if s != 0:
    #                 c[Trap, Left, s, t] = \
    #                       [c.merge(coder[Tri, Right, s:t, r], coder[Tri, Left, s+1:t+1, t])
    #                        out=[(t, s)])]

    #             c[Trap, Right, s, t] = \
    #                   [c.merge((Tri, Right, s, r), (Tri, Left, r+1, t),
    #                            out=[(s, t)])
    #                    for r in range(s, t)]

    #             if s != 0:
    #                 c[Tri, Left, s, t] = \
    #                     [c.merge((Tri, Left, s, r), (Trap, Left, r, t), out=[])
    #                      for r in range(s, t)]

    #             c[Tri, Right, s, t] = \
    #                   [c.merge((Trap, Right, s, r), (Tri, Right, r, t),
    #                            out=[])
    #                    for r in range(s + 1, t + 1)]

    #     hyper = c.finish(False)
    #     self.data = c.output_matrix

    #     return c


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
