import numpy as np

class IdHasher:
    def hash_item(self, i):
        return i

cdef class IndexSet:
    def __cinit__(self, size):

        if isinstance(size, int):
            self._hasher = None
            self._max_size = size
            return

        {% for i in range(2, 6) %}
        if len(size) == {{i}}:
            self._hasher = IntTuple{{i}}Hasher(*size)
            self._max_size = self._hasher.max_size()
        {% endfor %}

    def iter_items(self):
        cdef int i
        if self._hasher is None:
            for i in range(self._max_size):
                yield i, i
        else:
            for i in range(self._max_size):
                yield i, self._hasher.unhash(i)

    cpdef int index(self, element):
        if self._hasher is None: return element
        return self._hasher.hash_item(*element)

    cpdef element(self, int index):
        if self._hasher is None: return index
        return self._hasher.unhash(index)

    def elements(self, indices):
        cdef int i
        for i in indices:
            yield self._hasher.unhash(i)


    def item_vector(self, elements):
        data = []
        indices = []
        ind = [0]
        for element in elements:
            data.append(1)
            indices.append(self.index(element))
        ind.append(len(data))
        return scipy.sparse.csc_matrix(
            (data, indices, ind),
            shape=(self._max_size, 1),
            dtype=np.uint8)

    def __len__(self):
        return self._max_size

cdef class ChartEdge:
    pass

cdef class ChartBuilder:
    """
    A imperative interface for specifying dynamic programs.
    """

    def __init__(self,
                 debug=False,
                 strict=True,
                 IndexSet item_set=None,
                 IndexSet output_set = None,
                 expected_size=None):
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

        self._done = False
        self._last = -1
        self._debug = debug
        self._strict = strict
        self._hg_ptr = new CHypergraph(False)

        self._output_set = output_set
        self._item_set = item_set
        cdef size = len(self._item_set)
        self._chart = \
            new vector[const CHypernode*](size, NULL)
        self._label_chart = []
        if expected_size:
            self._hg_ptr.set_expected_size(size,
                                           expected_size[0],
                                           expected_size[1])
            self._max_arity = expected_size[1]

        self._data = []
        self._indices = []
        self._ind = [0]

        self._ndata = []
        self._nindices = []
        self._nind = [0]

    def __dealloc__(self):
        del self._chart

    property output_set:
        def __get__(self):
            return self._output_set

    property item_set:
        def __get__(self):
            return self._item_set

    def finish(self, reconstruct=True):
        """
        Finish the chart and get out the root value.
        """
        if self._done:
            raise Exception("Hypergraph not constructed.")
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

    def init(self): return ChartEdge()

    def merge(self, *args, values=[]):
        cdef ChartEdge chart_edge = ChartEdge()
        cdef int index
        if len(args) == 0:
            raise Exception("Merge takes at least one item.")

        for key in args:
            index = self._item_set.index(key)
            chart_edge.tail_ptrs.push_back(
                deref(self._chart)[index])
        chart_edge.values = values
        return chart_edge

    def __contains__(self, label):
        cdef int index = self._item_set.index(label)
        return deref(self._chart)[index] != NULL

    def __setitem__(self, key, chart_edges):
        cdef int index = self._item_set.index(key)
        if self._strict and deref(self._chart)[index] != NULL:
            raise Exception("Chart already has label %s"%(key,))
        if isinstance(chart_edges, ChartEdge):
            deref(self._chart)[index] = \
                self._hg_ptr.add_terminal_node()
            self._label_chart.append((index, key))

            self._ndata.append(1)
            self._nindices.append(index)
            self._nind.append(len(self._ndata))

            return

        deref(self._chart)[index] = self._hg_ptr.start_node()

        cdef ChartEdge chart_edge
        for chart_edge in chart_edges:
            edge_num = self._hg_ptr.add_edge(chart_edge.tail_ptrs)

            self._data += [1] * len(chart_edge.values)
            self._indices += [self._output_set.index(v)
                              for v in chart_edge.values]
            self._ind.append(len(self._data))

        result = self._hg_ptr.end_node()
        if not result:
            if self._strict:
                raise Exception("Index failed. %s"%(key,))
            deref(self._chart)[index] = NULL

        self._last = index
        self._label_chart.append((index, key))

        self._ndata.append(1)
        self._nindices.append(index)
        self._nind.append(len(self._ndata))

    def matrix(self):
        return scipy.sparse.csc_matrix(
            (self._data, self._indices, self._ind),
            shape=(len(self._output_set),
                   self._hg_ptr.edges().size()),
            dtype=np.uint8)

    def item_matrix(self):
        return scipy.sparse.csc_matrix(
            (self._ndata, self._nindices, self._nind),
            shape=(len(self._item_set),
                   self._hg_ptr.nodes().size()),
            dtype=np.uint8)




{% for i in range(2, 6) %}
cdef class IntTuple{{i}}:
    def __cinit__(self, {% for j in range(i) %}int {{var[j]}}, {% endfor %}
                  blank=None):
    {% for j in range(i) %}
        self.{{var[j]}} = {{var[j]}}
    {% endfor %}

    def unpack(self):
        return ({% for j in range(i) %} self.{{var[j]}}, {% endfor %})

cdef class IntTuple{{i}}Hasher:
    def __cinit__(self, {% for j in range(i) %}int {{var[j]}}, {% endfor %}):
        # n = {{i}}
        # m = ({% for j in range(i) %} {{var[j]}},{% endfor %})
        self._multipliers_a = 1
        {% for j in range(1,i) %}
        self._multipliers_{{var[j]}} = self._multipliers_{{var[j-1]}} * {{var[j-1]}}
        {% endfor %}
        self._max_size = {% for j in range(i) %} {{var[j]}} * {% endfor %} 1

    def max_size(self):
        return self._max_size

    cpdef unhash(self, int val):
        cdef t = []
        cdef int v = val
        {% for j in range(i-1, -1, -1) %}
        #for k in range({{i}}-1, -1, -1):
        t.insert(0, v / self._multipliers_{{var[j]}})
        v = v % self._multipliers_{{var[j]}}
        {% endfor %}
        return tuple(t)



    cpdef int hash_item(self, {% for j in range(i) %}int {{var[j]}}, {% endfor %}):
        return \
          {% for j in range(i) %} \
            {{var[j]}} * self._multipliers_{{var[j]}} + \
          {% endfor %} + 0
{% endfor %}

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

    cpdef int hash_item(self, t):
        val = self._multipliers[0] * t[0]
        cdef int multiplier = 1
        for i in range(1, len(t)):
            val += self._multipliers[i] * t[i]
        return val
