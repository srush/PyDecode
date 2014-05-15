class DependencyParse:
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

    @staticmethod
    def enumerate_projective(n, m=None):
        """
        Enumerate all possible projective trees.

        Parameters
        ----------
        n - int
           Length of sentence (without root symbol)

        m - int
           Number of modifiers to use.

        Returns
        --------
        parses : iterator
           An iterator of possible (m,n) parse trees.
        """
        for mid in itertools.product([None] + range( n + 1), repeat=n):
            parse = Parse([-1] + list(mid))
            if m is not None and parse.skipped_words() != n- m:
                continue

            if (not parse.check_projective()) or (not parse.check_spanning()): 
                continue
            yield parse


# Globals
Tri = 1
TrapSkipped = 2
Trap = 3
Box = 4

Right = 0
Left = 1

def NodeType(type, dir, span, count, states=None) :
    if states is None:
        return (type, dir, span, count)
    return (type, dir, span, count, states)
def node_type(nodetype): return nodetype[0]
def node_span(nodetype): return nodetype[2]
def node_dir(nodetype): return nodetype[1]

class Arc(namedtuple("Arc", ["head", "mod", "sibling"])):
    def __new__(cls, head, mod, sibling=None):
        return super(Arc, cls).__new__(cls, head, mod, sibling)

def parse_first_order(self, sentence_length, chart=None):
    """
    First-order dependency parsing.

    Parameters
    -----------
    sentence_length : int
       The length of the sentence.

    Returns
    -------
    chart : 
       The finished chart.
    """
    n = sentence_length + 1

    # Add terminal nodes.
    [c.init(NodeType(sh, d, (s, s), 0), 0.0)
     for s in range(n)
     for d in [Right, Left]
     for sh in [Trap, Tri]]

    for k in range(1, n):
        for s in range(n):
            t = k + s
            if t >= n: break
            span = (s, t)
            need = m

            # First create incomplete items.
            if s != 0:
                c[NodeType(Trap, Left, span)] = \
                    c.sum([key1 * key2 * Arc(t, s)
                           for r in range(s, t)
                           for key1 in [(Tri, Right, (s, r))]
                           if key1 in c.chart
                           for key2 in [(Tri, Left, (r+1, t))]
                           if key2 in c.chart
                           ])

            
            c[NodeType(Trap, Right, span)] = \ 
                c.sum([ key1 * key2 * Arc(s, t)
                       for r in range(s, t)
                       for key1 in [(Tri, Right, (s, r))]
                       if key1 in c.chart
                       for key2 in [(Tri, Left, (r+1, t))]
                       if key2 in c.chart])

            if s != 0:
                c[NodeType(Tri, Left, span)]= \ 
                c.sum([key1 * key2
                       for r in range(s, t)
                       for key1 in [(Tri, Left, (s, r))]
                       if key1 in c.chart
                       for key2 in [(Trap, Left, (r, t))]
                       if key2 in c.chart])

            c[NodeType(Tri, Right, span)] = \
                c.sum(key1 * key2
                      for r in range(s + 1, t + 1)
                      for key1 in [(Trap, Right, (s, r))]
                      if key1 in c.chart
                      for key2 in [(Tri, Right, (r, t))]
                      if key2 in c.chart]

    return make_parse(n, c.backtrace(NodeType(Tri, Right, (0, n-1))))
