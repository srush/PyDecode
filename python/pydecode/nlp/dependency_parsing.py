"""
Classes for dependency parsing problems.
"""
import pydecode
import numpy as np
import itertools
from pydecode.encoder import StructuredEncoder


class DependencyParsingEncoder(StructuredEncoder):
    def __init__(self, n, order):
        self.n = n
        self.order = order
        if order == 1:
            shape = (n, n)
        elif order == 2:
            shape = (n, n, n)
        super(DependencyParsingEncoder, self).__init__(shape)

    def transform_structure(self, parse):
        if self.order == 1:
            return np.array([[h, m]
                             for m, h in enumerate(parse[1:], 1)])
        elif self.order == 2:
            return np.array([[h, m, sibling(parse, m)]
                             for m, h in enumerate(parse[1:], 1)])

    def from_parts(self, parts):
        arcs = list(parts)
        arcs.sort(key=lambda a:a[1])
        return np.array(([-1] + [arc[0] for arc in arcs if arc[1] != 0]))

    def all_structures(self):
        n = self.n
        for mid in itertools.product(range(n+1), repeat=n-1):
            parse = np.zeros(n)
            parse[0] = -1
            parse[1:] = list(mid)
            if (not is_projective(parse)) or \
                    (not is_spanning(parse)):
                continue
            yield parse

    def random_structure(self):
        n = self.n
        while True:
            mid = np.random.randint(n+1, size=n-1)
            parse = np.zeros(n)
            parse[0] = -1
            parse[1:] = mid
            if (not is_projective(parse)) or \
                    (not is_spanning(parse)):
                continue
            return parse


# Globals
kShapes = 3
Tri = 0
Trap = 1
Box = 2

kDir = 2
Right = 0
Left = 1

def eisner_first_order(n):
    num_edges = 4 * n ** 3

    items = np.arange((kShapes * kDir * n * n), dtype=np.int64) \
        .reshape([kShapes, kDir, n, n])
    part_encoder = DependencyParsingEncoder(n, 1)
    out = part_encoder.encoder
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
                c.set_t(items[Trap, Left,  s,       t],
                        items[Tri,  Right, s,       s:t],
                        items[Tri,  Left,  s+1:t+1, t],
                        labels=out_ind)

            out_ind.fill(out[s, t])
            c.set_t(items[Trap, Right, s,       t],
                    items[Tri,  Right, s,       s:t],
                    items[Tri,  Left,  s+1:t+1, t],
                    labels=out_ind)

            out_ind.fill(-1)
            if s != 0:
                c.set_t(items[Tri,  Left,  s,   t],
                        items[Tri,  Left,  s,   s:t],
                        items[Trap, Left,  s:t, t],
                        labels=out_ind)

            c.set_t(items[Tri,  Right, s,       t],
                    items[Trap, Right, s,       s+1:t+1],
                    items[Tri,  Right, s+1:t+1, t],
                    labels=out_ind)

    return c.finish(False), part_encoder


def eisner_second_order(n):
    coder = np.arange((kShapes * kDir * n * n), dtype=np.int64) \
        .reshape([kShapes, kDir, n, n])

    part_encoder = DependencyParsingEncoder(n, 2)
    out = part_encoder.encoder

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
                c.set_t(coder[Box, Left, s, t],
                        coder[Tri, Right, s, s:t],
                        coder[Tri, Left, s+1:t+1, t])

                c.set_t(coder[Trap, Left, s, t],
                        np.append(coder[Tri, Right, s, t-1],
                                  coder[Box, Left, s, t-1:s:-1]),
                        np.append(coder[Tri, Left, t, t],
                                  coder[Trap, Left, t-1:s:-1, t]),
                        labels=out[t, s, t:s:-1])

            c.set_t(coder[Trap, Right, s, t],
                    np.append(coder[Tri, Right, s, s],
                              coder[Trap, Right, s, s+1:t]),
                    np.append(coder[Tri, Left, s+1, t],
                              coder[Box, Left, s+1:t, t]),
                    labels=out[s, t, s:t])

            if s != 0:
                c.set_t(coder[Tri, Left, s, t],
                        coder[Tri, Left, s, s:t],
                        coder[Trap, Left, s:t, t])

            if (s, t) == (0, n-1) or s != 0:
                c.set_t(coder[Tri, Right, s, t],
                        coder[Trap, Right, s, s+1:t+1],
                        coder[Tri, Right, s+1:t+1, t])
    return c.finish(False), part_encoder


def is_spanning(parse):
    """
    Is the parse tree as valid spanning tree?

    Returns
    --------
    spanning : bool
    True if a valid spanning tree.
    """
    d = {}
    for m, h in enumerate(parse):
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
    if len(seen) != len(parse) - \
            len([1 for p in parse if p is None]):
        return False
    return True

def is_projective(parse):
    """
    Is the parse tree projective?

    Returns
    --------
    projective : bool
       True if a projective tree.
    """
    for m, h in enumerate(parse):
        for m2, h2 in enumerate(parse):
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

def siblings(parse, m):
    return [m2 for (m2, h) in enumerate(parse)
            if h == parse[m]
            if m != m2]

def sibling(parse, m):
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
    sibs = siblings(parse, m)
    h = self.heads[m]
    if m > h:
        return max([s2 for s2 in sibs if h < s2 < m] + [h])
    if m < h:
        return min([s2 for s2 in sibs if h > s2 > m] + [h])
