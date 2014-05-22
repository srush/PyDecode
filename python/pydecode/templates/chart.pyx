# class HypergraphSemiRing:
#     def __init__(self, name=None,
#                  edge_list=[], node_list=[], is_zero=False):
#         self.edge_list = edge_list
#         self.node_list = node_list
#         self.name = name
#         self._is_zero = is_zero

#     @staticmethod
#     def from_value(name):
#         return HypergraphSemiRing(name)

#     def __repr__(self):
#         return "%s %s %s %s" % (self.edge_list,
#                                 self.node_list,
#                                 self.name,
#                                 self._is_zero)

#     def is_zero(self):
#         return self._is_zero

#     def __add__(self, other):
#         edges = self.edges() + other.edges()
#         return HypergraphSemiRing(name=None,
#                                   edge_list=edges)

#     def edges(self):
#         if self.node_list:
#             return self.edge_list + [(self.node_list, self.name)]
#         else:
#             return self.edge_list

#     def __mul__(self, other):
#         zero = other.is_zero() or self.is_zero()
#         if zero:
#             return HypergraphSemiRing.zero()
#         return HypergraphSemiRing(name=other.name, edge_list=[],
#                                   node_list=self.node_list + other.node_list)

#     @classmethod
#     def one(cls):
#         return HypergraphSemiRing()

#     @classmethod
#     def zero(cls):
#         return HypergraphSemiRing(is_zero=True)


cdef class DPChartBuilder:
    """
    A imperative interface for specifying dynamic programs.
    """

    def __init__(self, score_fn=None,
                 semiring=None,
                 build_hypergraph=False,
                 debug=False,
                 strict=True,
                 hasher=None):
        """
        Initialize the dynamic programming chart.

        Parameters
        ------------

        score_fn :  label -> "score"
           A function from edge label to score.

        semiring : :py:class:`SemiRing`
           The semiring class to use.

        build_hypergraph : bool
           Should we build a hypergraph in the chart.

        """

        self._semiring = semiring
        self._scorer = score_fn
        self._done = False
        self._last = -1
        self._debug = debug
        self._strict = strict
        self._hasher = lambda a: a
        self._hg_ptr = new CHypergraph(False)

    def __dealloc__(self):
        del self._chart

    def set_hasher(self, hasher):
        self._hasher = hasher
        self._chart = new vector[const CHypernode*](self._hasher.max_size(), NULL)
        self._label_chart = []

    def finish(self):
        """
        Finish the chart and get out the root value.
        """
        if self._done:
            raise Exception("Hypergraph not constructed")
        self._done = True
        self._hg_ptr.finish()

        final_node_labels = [None] * self._hg_ptr.nodes().size()
        for h, label in self._label_chart:
            if deref(self._chart)[h] != NULL and deref(self._chart)[h].id() >= 0:
                final_node_labels[deref(self._chart)[h].id()] = label

        hypergraph = Hypergraph(False)
        hypergraph.init(self._hg_ptr, Labeling(hypergraph, final_node_labels))
        return hypergraph

    def init(self, label):
        """
        Initialize a chart cell to the 1 value.

        Parameters
        ------------

        label : any
           The node to initialize.
        """
        h = self._hasher(label)
        deref(self._chart)[h] = self._hg_ptr.add_terminal_node()
        self._label_chart.append((h, label))

    def set(self, label, val):
        h = self._hasher(label)
        cdef vector[const CHypernode *] tail_node_ptrs

        if self._strict and label in self:
            raise Exception(
                "Chart already has label {}".format(label))

        deref(self._chart)[h] = self._hg_ptr.start_node()
        for tail_node_keys, edge_label in val:
            if len(tail_node_keys) == 0:
                raise HypergraphConstructionException(
                    "An edge must have at least one tail node.")
            tail_node_ptrs.clear()
            for tail_node_key in tail_node_keys:
                tail_node_ptrs.push_back(deref(self._chart)[self._hasher(tail_node_key)])
            self._hg_ptr.add_edge(tail_node_ptrs)

        result = self._hg_ptr.end_node()
        if not result:
            deref(self._chart)[h] = NULL

        self._last = h
        self._label_chart.append((h, label))

    def __contains__(self, label):
        h = self._hasher(label)
        return deref(self._chart)[h] != NULL



cdef class SizedTupleHasher:
    """
    For hashing chart items to integers.
    """
    def __init__(self, sizes):
        cdef int multi = 1
        for s in sizes:
            self._multipliers.push_back(multi)
            multi *= s
        self._max_size = multi

    def __call__(self, tupl):
        cdef int val = 0
        cdef int multiplier = 1
        for i in range(len(tupl)):
            val += self._multipliers[i] * tupl[i]
        return val

    def max_size(self):
        return self._max_size

    def mult_size(self):
        return self._multipliers.size()

    def unhash(self, int val):
        cdef t = []
        cdef int v = val
        for i in range((<int>self._multipliers.size())-1, -1, -1):
            t.insert(0, v / self._multipliers[i])
            v = v % self._multipliers[i]
        return tuple(t)
