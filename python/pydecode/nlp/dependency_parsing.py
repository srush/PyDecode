"""
Classes for dependency parsing problems.
"""

import pydecode.nlp.decoding as decoding
import pydecode
import numpy.random
import numpy as np
import itertools

class DependencyProblem(decoding.DecodingProblem):
    """
    Descriptions for dependency parsing problem over a
    sentence x_1 ... x_n, where x_0 is an implied root
    vertex.
    """
    def __init__(self, size):
        """
        Parameters
        ----------
        size : int
           The length of the sentence n.
        """
        self.size = size

    def feasible_set(self):
        """
        Generate all possible projective trees.

        Returns
        --------
        parses : iterator
           An iterator of possible n-parse trees.
        """
        n = self.size
        for mid in itertools.product(range(n + 1), repeat=n):
            parse = DependencyParse([-1] + list(mid))
            if (not parse.check_projective()) or \
                    (not parse.check_spanning()):
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

    def arcs(self):
        """
        Returns
        -------
        arc : iterator of (m, h) pairs in m order
           Each of the arcs used in the parse.
        """
        for m, h in enumerate(self.heads):
            if m == 0 or h is None:
                continue
            yield (m, h)

    def siblings(self, m):
        return [m2 for (m2, h) in self.arcs()
                if h == self.heads[m]
                if m != m2]

    def sibling(self, m):
        """
        Parameters
        ----------
        m : int
           Sentence position in {1..n}.

        Returns
        -------
        sibling : int
           The sibling of m in the parse.
        """
        sibs = self.siblings(m)
        h = self.heads[m]
        if m > h:
            return max([s2 for s2 in sibs if h < s2 < m] + [h])
        if m < h:
            return min([s2 for s2 in sibs if h > s2 > m] + [h])


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
        if len(seen) != len(self.heads) - \
                len([1 for p in self.heads if p is None]):
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
                if m2 == m:
                    continue
                if m < h:
                    if m < m2 < h < h2 or m < h2 < h < m2 or \
                            m2 < m < h2 < h or  h2 < m < m2 < h:
                        return False
                if h < m:
                    if h < m2 < m < h2 or h < h2 < m < m2 or \
                            m2 < h < h2 < m or  h2 < h < m2 < m:
                        return False
        return True

class FirstOrderCoder(object):
    """
    Bijective map between DependencyParse and sparse output
    representation as arcs (h, m).
    """
    def __init__(self, problem):
        self._problem = problem
        n = problem.size
        self.shape_ = [n+1, n+1]

    def inverse_transform(self, arcs):
        """
        Map sparse output to DependencyParse.
        """
        arcs = list(arcs)
        arcs.sort(key=lambda a:a[1])
        heads = ([-1] + [head for head, m in arcs if m != 0])
        return DependencyParse(heads)

    def transform(self, parse):
        """
        Map DependencyParse to sparse output.
        """
        return np.array([[h, m] for m, h in parse.arcs()])

class SecondOrderCoder(object):
    """
    Bijective map between DependencyParse and sparse output
    representation as arcs (h, m, s).
    """
    def __init__(self, problem):
        self._problem = problem
        n = self._problem.size
        self.shape_ = [n+1, n+1, n+1]

    def inverse_transform(self, arcs):
        """
        Map sparse output to dependency parse.
        """
        arcs = list(arcs)
        arcs.sort(key=lambda a:a[1])
        heads = ([-1] + [head for head, m, _ in arcs if m != 0])
        return DependencyParse(heads)

    def transform(self, parse):
        """
        Map DependencyParse to sparse output.
        """
        return np.array([[h, m, parse.sibling(m)]
                         for m, h in parse.arcs()])

# Globals
kShapes = 3
Tri = 0
Trap = 1
Box = 2

kDir = 2
Right = 0
Left = 1

class FirstOrderDecoder(decoding.HypergraphDecoder):
    def output_coder(self, problem):
        return FirstOrderCoder(problem)

    def dynamic_program(self, problem):
        """
        Implements Eisner's algorithm for first-order parsing.

        Parameters
        -----------
        problem : DependencyProblem
          Problem description.

        Returns
        -------
        dp : DynamicProgram
        """
        n = problem.size + 1
        num_edges = 4 * n ** 3

        items = np.arange((kShapes * kDir * n * n), dtype=np.int64) \
            .reshape([kShapes, kDir, n, n])
        out = np.arange(n*n, dtype=np.int64).reshape([n, n])
        c = pydecode.ChartBuilder(items, out,
                                  unstrict=True,
                                  expected_size=(num_edges, 2))

        # Add terminal nodes.
        c.init(np.diag(items[Tri, Right]).copy())
        c.init(np.diag(items[Tri, Left, 1:, 1:]).copy())

        for k in range(1, n):
            for s in range(n):
                t = k + s
                if t >= n:
                    break

                out_ind = np.zeros([t-s], dtype=np.int64)

                # First create incomplete items.
                if s != 0:
                    out_ind.fill(out[t, s])
                    c.set(items[Trap, Left,  s,       t],
                           items[Tri,  Right, s,       s:t],
                           items[Tri,  Left,  s+1:t+1, t],
                           out=out_ind)

                out_ind.fill(out[s, t])
                c.set(items[Trap, Right, s,       t],
                      items[Tri,  Right, s,       s:t],
                      items[Tri,  Left,  s+1:t+1, t],
                      out=out_ind)

                out_ind.fill(-1)
                if s != 0:
                    c.set(items[Tri,  Left,  s,   t],
                          items[Tri,  Left,  s,   s:t],
                          items[Trap, Left,  s:t, t],
                          out=out_ind)

                c.set(items[Tri,  Right, s,       t],
                      items[Trap, Right, s,       s+1:t+1],
                      items[Tri,  Right, s+1:t+1, t],
                      out=out_ind)

        return c.finish(False)

class SecondOrderDecoder(decoding.HypergraphDecoder):
    def output_coder(self, problem):
        return SecondOrderCoder(problem)

    def dynamic_program(self, problem):
        """
        Implements Eisner's algorithm for second-order parsing.

        Parameters
        -----------
        problem : DependencyProblem
          Problem description.

        Returns
        -------
        dp : DynamicProgram
        """
        n = problem.size + 1

        coder = np.arange((kShapes * kDir * n * n), dtype=np.int64) \
            .reshape([kShapes, kDir, n, n])
        out = np.arange(n*n*n, dtype=np.int64).reshape([n, n, n])
        c = pydecode.ChartBuilder(coder, out,
                                  unstrict=True)
        # Initialize the chart.
        c.init(np.diag(coder[Tri, Right]).copy())
        c.init(np.diag(coder[Tri, Left, 1:, 1:]).copy())

        for k in range(1, n):
            for s in range(n):
                t = k + s
                if t >= n:
                    break

                if s != 0:
                    c.set(coder[Box, Left, s, t],
                          coder[Tri, Right, s, s:t],
                          coder[Tri, Left, s+1:t+1, t])

                    c.set(coder[Trap, Left, s, t],
                          np.append(coder[Tri, Right, s, t-1],
                                    coder[Box, Left, s, t-1:s:-1]),
                          np.append(coder[Tri, Left, t, t],
                                    coder[Trap, Left, t-1:s:-1, t]),
                          out=out[t, s, t:s:-1])

                c.set(coder[Trap, Right, s, t],
                      np.append(coder[Tri, Right, s, s],
                                coder[Trap, Right, s, s+1:t]),
                      np.append(coder[Tri, Left, s+1, t],
                                coder[Box, Left, s+1:t, t]),
                      out=out[s, t, s:t])

                if s != 0:
                    c.set(coder[Tri, Left, s, t],
                          coder[Tri, Left, s, s:t],
                          coder[Trap, Left, s:t, t])

                if (s, t) == (0, n-1) or s != 0:
                    c.set(coder[Tri, Right, s, t],
                          coder[Trap, Right, s, s+1:t+1],
                          coder[Tri, Right, s+1:t+1, t])
        return c.finish(False)
