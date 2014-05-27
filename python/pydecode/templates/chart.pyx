import numpy as np

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
        self._chart = NULL

    def __dealloc__(self):
        del self._chart

    def set_hasher(self, hasher):
        self._hasher = hasher
        self._chart = \
            new vector[const CHypernode*](self._hasher.max_size(), NULL)
        self._label_chart = []

    def set_expected_size(self, int num_vertices, int num_edges, int max_arity):
        self._hg_ptr.set_expected_size(num_vertices, num_edges, max_arity)
        self._max_arity = max_arity

    def set_data(self, data):
        self._data = data

    def finish(self, reconstruct=True):
        """
        Finish the chart and get out the root value.
        """
        if self._done:
            raise Exception("Hypergraph not constructed")
        self._done = True
        self._hg_ptr.finish(reconstruct)


        final_node_labels = [None] * self._hg_ptr.nodes().size()
        for h, label in self._label_chart:
            if deref(self._chart)[h] != NULL and deref(self._chart)[h].id() >= 0:
                final_node_labels[deref(self._chart)[h].id()] = label

        hypergraph = Hypergraph(False)
        hypergraph.init(self._hg_ptr,
                        Labeling(hypergraph, final_node_labels))
        return hypergraph

    def init(self, label):
        """
        Initialize a chart cell to the 1 value.

        Parameters
        ------------

        label : any
           The node to initialize.
        """
        h = self._hasher.hasher(label)
        deref(self._chart)[h] = self._hg_ptr.add_terminal_node()
        self._label_chart.append((h, label))


    def set(self, node_label, val):
        h = self._hasher.hasher(node_label)
        cdef vector[const CHypernode *] tail_node_ptrs

        if self._strict and node_label in self:
            raise Exception(
                "Chart already has label")

        deref(self._chart)[h] = self._hg_ptr.start_node()
        for tail_node_keys, edge_label in val:
            if len(tail_node_keys) == 0:
                raise Exception(
                    "An edge must have at least one tail node.")

            tail_node_ptrs.clear()
            for j, tail_node_key in enumerate(tail_node_keys):
                tail_node_ptrs.push_back(
                    deref(self._chart)[self._hasher.hasher(tail_node_key)])

            edge_num = self._hg_ptr.add_edge(tail_node_ptrs)
            for i, label in enumerate(edge_label):
                self._data[i][edge_num] = label

        result = self._hg_ptr.end_node()
        if not result:
            deref(self._chart)[h] = NULL

        self._last = h
        self._label_chart.append((h, node_label))

    def __contains__(self, label):
        h = self._hasher.hasher(label)
        return deref(self._chart)[h] != NULL



cdef class Quartet:
    def __cinit__(self, int a, int b, int c, int d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def unpack(self):
        return (self.a, self.b, self.c, self.d)

cdef class QuartetHash:
    def __cinit__(self, Quartet t):
        self._multipliers = Quartet(1, t.a, t.a * t.b, t.a * t.b * t.c)
        self._max_size = t.a * t.b * t.c * t.d

    def max_size(self):
        return self._max_size

    cpdef int hasher(self, Quartet t):
        return t.a * self._multipliers.a + t.b * \
            self._multipliers.b + t.c * \
            self._multipliers.c + t.d * self._multipliers.d

    def __dealloc__(self):
        pass
        # val = self._multipliers[0] * t[0]

        # cdef int multiplier = 1
        # for i in range(1, len(t)):
        #     val += self._multipliers[i] * t[i]
        # return val


cdef class SizedTupleHasher:
    """
    For hashing chart items to integers.
    """
    def __init__(self, sizes):
        cdef int multi = 1
        multipliers = []
        for s in sizes:
            self._multipliers.push_back(multi)
            multipliers.append(multi)
            multi *= s
        self._max_size = multi
        self._np_multi = np.array(multipliers)

    def max_size(self):
        return self._max_size

    cpdef hasher(self, t):
        val = self._multipliers[0] * t[0]
        cdef int multiplier = 1
        for i in range(1, len(t)):
            val += self._multipliers[i] * t[i]
        return val

    # def hash_mat(self, t):
    #     return t.dot(self._np_multi)


    # def mult_size(self):
    #     return self._multipliers.size()

    # def unhash(self, int val):
    #     cdef t = []
    #     cdef int v = val
    #     for i in range((<int>self._multipliers.size())-1, -1, -1):
    #         t.insert(0, v / self._multipliers[i])
    #         v = v % self._multipliers[i]
    #     return tuple(t)

    # def hash_template(self, template):
    #     cdef int val = np.zeros()
    #     cdef int multiplier = 1
    #     for i in range(len(template)):
    #         if isinstance(template[i], int):
    #             val += self._multipliers[i] * tupl[i]
    #     return val
