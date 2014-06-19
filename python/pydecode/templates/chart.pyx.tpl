import numpy as np
cimport cython

class IdHasher:
    def hash_item(self, i):
        return i

cdef class IndexedEncoder:
    """
    Encoder mapping integer tuples to integers.

    Encodes the mapping N_1 x N_2 x ... N_k -> {0...|N_1||N_2|...|N_k|}.
    Attributes
    ----------
    max_size : int
        The range size |N_1||N_2|...|N_k|.

    """
    def __cinit__(self, shape):
        """
        Initialize the encoder from set sizes.

        Parameters
        -----------
        size : int tuple
           A tuple of (|N_1|, |N_2|, ..., |N_k|)

        """
        self._multipliers = np.zeros([len(shape), 1], dtype=np.int32)
        self._shape = shape
        self._multipliers[0] = 1

        for j in range(1, len(shape)):
            self._multipliers[j] = self._multipliers[j-1] * shape[j-1]
        self._max_size = np.product(shape)

    cpdef np.ndarray transform(self, np.ndarray element):
        """
        Transform from tuple to int.
        """
        return np.dot(self._multipliers.T, element.T)

    cpdef np.ndarray inverse_transform(self, np.ndarray indices):
        """
        Inverse transform from int to tuple.
        """
        v = indices
        m = np.zeros((len(self._multipliers), len(indices)), dtype=np.int32)
        for j in range(len(self._multipliers) - 1, -1, -1):
            m[j, :] =  v // self._multipliers[j]
            v = v % self._multipliers[j]
        return m

    def reshape(self, matrix):
        assert matrix.shape == self._shape
        return np.reshape(matrix.T, (self.max_size, 1))

#         self._multipliers[0] = 1

#         for j in range(1, len(size)):
#             self._multipliers[j] = self._multiplier[j-1] * size[j-1]

    property max_size:
        def __get__(self):
            return self._max_size

#     cpdef int [:] transform(self, int [:,:] element):
#         """
#         Transform from tuple to int.
#         """
#         return self._multipliers * element

#     cpdef int [:,:] inverse_transform(self, int [:]  index):
#         """
#         Inverse transform from int to tuple.
#         """
#         if self._hasher is None: return index
#         return self._hasher.unhash(index)


#     def iteritems(self):
#         cdef int i
#         if self._hasher is None:
#             for i in range(self._max_size):
#                 yield i, i
#         else:
#             for i in range(self._max_size):
#                 yield i, self._hasher.unhash(i)


    def item_vector(self, elements):
        data = []
        indices = []
        ind = [0]
        for element in elements:
            data.append(1)
            indices.append(self.transform(element))
        ind.append(len(data))
        return scipy.sparse.csc_matrix(
            (data, indices, ind),
            shape=(self._max_size, 1),
            dtype=np.uint8)


cdef class _ChartEdge:
    pass

cdef class ChartBuilder:
    """
    ChartBuilder is an interface for specifying dynamic programs.

    The chart acts like a dictionary between items I and "tokens". ::
       >> c[item] = c.init()
       >> c[item2] = [c.merge(item)]
       >> c[item3] = [c.merge(item, item2), c.merge(item)]
       >> c[item4] = [c.merge(item3, out=[output])]

    When a chart is complete, it creates a hypergraph (V, E).

       >> hypergraph = c.finish()

    The chart builder also maintains a mapping between the hypergraph
    and item set and output set.

    Define the set of items I to specify the cells in a dynamic
    programming chart.

    Define the set of outputs O to specify the output emitted by a
    decision in the dynamic program.

    Attributes
    ----------
    item_encoder : Encoder I -> {0...|I|}
       Encodes elements of the item set I as integers.

    output_encoder : Encoder O -> {0...|O|}
       Encodes elements of the output set O as integers.

    item_matrix : Sparse matrix |I| x |V|
       Maps vertices of the hypergraph to item indicators.

    output_matrix : Sparse matrix |O| x |E|
       Maps hyperedges of the hypergraph to sparse output counts.
    """

    def __init__(self,
                 item_encoder,
                 item_set_size=None,
                 output_encoder=None,
                 output_size=None,
                 unstrict=False,
                 expected_size=None):
        """
        Initialize the dynamic programming chart.

        Parameters
        ------------

        item_encoder, item_set_size : Encoder I -> {0...|I|}, Int
            Specifies the item set I for chart, and the size |I|
            The encoder must have a `transform` method.

        output_encoder : Encoder O -> {0...|O|}
            Specifies the item set O for chart.
            The encoder must have a `transform` method.

        unstrict : bool
            Allows the chart to merge NULL items.

        expected_size : (int, int)
            Set the expected number of edges |E| and
            the max-arity of edges. Useful for efficiency.
        """

        self._done = False
        self._last = -1
        self._strict = not unstrict
        self._hg_ptr = new CHypergraph(False)

        self._item_encoder = item_encoder
        cdef int size = item_set_size
        self._chart = new vector[const CHypernode *](size, NULL)
        self._item_size = size

        self._output_encoder = output_encoder
        self._output_size = output_size

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

    # def init(self):
    #     """
    #     Returns the initial value for a chart item. Usage::

    #        >> c[item] = c.init()
    #     Returns
    #     --------
    #      : token
    #        Chart token used to initialize a cell.

    #     """
    #     return _ChartEdge()

    # cpdef _ChartEdge merge():
    #     cdef _ChartEdge chart_edge = _ChartEdge()

    def merge(self, *args, out=[]):
        """
        Merges the items given as arguments. Usage::

           >> c[item_head] = [c.merge(item_tail1, item_tail2, out=[2])]

        Parameters
        ----------
        *args : list of items in I
            The items to merge.

        out : list of outputs in O
            The outputs to associate with this merge.

        Returns
        --------
         : token
           Chart token used to represent the merge.

        """

        cdef int index
        if len(args) == 0:
            raise Exception("Merge takes at least one item.")
        # chart_edge.items = np.array(args)
        return self._merge(args, out)

    @cython.boundscheck(False)
    cdef _ChartEdge  _merge(self, args, outs):
        cdef _ChartEdge chart_edge = _ChartEdge()
        chart_edge.tail_ptrs.resize(len(args))
        cdef int i = 0
        for arg in args:
            chart_edge.tail_ptrs[i] = deref(self._chart)[
                    self._item_encoder[arg]]
            i += 1
        chart_edge.values = [self._output_encoder[out] for out in outs]
        return chart_edge

    def finish(self, reconstruct=True):
        """
        Finish chart construction.

        Returns
        -------
        hypergraph :
           The hypergraph representing the dynamic program.

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

    def __contains__(self, item):
        cdef int index = self._item_encoder.transform(np.array([item]))[0,0]
        return deref(self._chart)[index] != NULL

    # def build_up(self, chart_edges):
    #     cdef _ChartEdge chart_edge

    #     cdef np.ndarray[np.int_t, ndim=2] stack1 = np.zeros([len(chart_edges), 4],
    #                                                        dtype=np.int)
    #     cdef np.ndarray[np.int_t, ndim=2] stack2 = np.zeros([len(chart_edges), 4],
    #                                                        dtype=np.int)
    #     cdef np.ndarray[np.int_t, ndim=2] values = np.zeros([len(chart_edges), 2],
    #                                                        dtype=np.int)

    #     cdef int j = 0
    #     cdef int i
    #     cdef np.ndarray[np.int_t, ndim=2] items
    #     for i, chart_edge in enumerate(chart_edges):
    #         items = chart_edge.items
    #         stack1[i,:] = items[0]
    #         stack2[i,:] = items[1]
    #         for v in chart_edge.values:
    #             values[j,:] = v
    #             j+=1


        # ind1 = self._item_encoder.transform(stack1)[0]
        # ind2 = self._item_encoder.transform(stack2)[0]
        # trans_values = self._output_encoder.transform(values)[0]
        # return ind1, ind2, trans_values

    # def build_up2(self, edges1, edges2, values):
    #     cdef _ChartEdge chart_edge

        # cdef np.ndarray[np.int_t, ndim=2] stack1 = np.zeros([len(edges1), 4],
        #                                                    dtype=np.int)
        # cdef np.ndarray[np.int_t, ndim=2] stack2 = np.zeros([len(edges2), 4],
        #                                                    dtype=np.int)
        # cdef np.ndarray[np.int_t, ndim=2] values_a = np.zeros([len(values), 2],
        #                                                    dtype=np.int)

        # cdef int j = 0
        # cdef int i
        # cdef np.ndarray[np.int_t, ndim=2] items
        # for i, edge in enumerate(edges1):
        #     stack1[i,:] = edges1[i]
        #     stack2[i,:] = edges2[i]
            # for v in chart_edge.values:
            #     values[j,:] = v
            #     j+=1


        # ind1 = self._item_encoder.transform(edges1)[0]
        # ind2 = self._item_encoder.transform(edges2)[0]
        # trans_values = self._output_encoder.transform(values)[0]
        # return ind1, ind2, trans_values

    # def __setitem__(self, key, q):

    #     cdef int index = self._item_encoder.transform(np.array([key]))[0,0]
    #     if self._strict and deref(self._chart)[index] != NULL:
    #         raise Exception("Chart already has label %s"%(key,))
    #     if isinstance(q, _ChartEdge):
    #         deref(self._chart)[index] = self._hg_ptr.add_terminal_node()
    #         self._label_chart.append((index, key))

    #         self._ndata.append(1)
    #         self._nindices.append(index)
    #         self._nind.append(len(self._ndata))
    #         return
    #     arr = q
    #     values = np.array([[0,0]])
    #     deref(self._chart)[index] = self._hg_ptr.start_node()

    #     ind1, ind2, trans_values = self.build_up2(arr[:,:4], arr[:,4:], values)

    #     cdef int j = 0
    #     cdef int i
    #     cdef vector[const CHypernode *] tail_ptrs
    #     #cdef _ChartEdge chart_edge
    #     for i in range(len(ind1)):
    #         tail_ptrs.clear()
    #         tail_ptrs.push_back(deref(self._chart)[ind1[i]])
    #         tail_ptrs.push_back(deref(self._chart)[ind2[i]])

    #         edge_num = self._hg_ptr.add_edge(tail_ptrs)

    #         # for v in chart_edge.values:
    #         #     self._data.append(1)
    #         #     self._indices.append(trans_values[j])
    #         #     j += 1
    #         # self._ind.append(len(self._data))

    #     result = self._hg_ptr.end_node()
    #     if not result:
    #         if self._strict:
    #             raise Exception("Index failed. %s"%(key,))
    #         deref(self._chart)[index] = NULL

    #     self._last = index
    #     self._label_chart.append((index, key))

    #     self._ndata.append(1)
    #     self._nindices.append(index)
    #     self._nind.append(len(self._ndata))

    @cython.boundscheck(False)
    cpdef init(self, int index):
        deref(self._chart)[index] = self._hg_ptr.add_terminal_node()
        # self._label_chart.append((index, key))

        self._ndata.append(1)
        self._nindices.append(index)
        self._nind.append(len(self._ndata))


    @cython.boundscheck(False)
    cpdef set2(self, long key, long [:] edges1, long [:] edges2,
               long [:] labels):
        cdef long index = key
        deref(self._chart)[index] = self._hg_ptr.start_node()
        cdef vector[const CHypernode *] tails
        J = edges1.shape[0]
        cdef int i, j
        for i in range(2):
            tails.clear()
            for j in range(J):
                if i == 0:
                    tails.push_back(deref(self._chart)[edges1[j]])
                elif i == 1:
                    tails.push_back(deref(self._chart)[edges2[j]])
            edge_num = self._hg_ptr.add_edge(tails)

        # for i in range(labels.shape[0]):
        #     self._ind.append(labels[i])
        #     self._data.append(1)
        #     self._indices.append(len(self._data))


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

        # cdef int index = self._item_encoder[key]
        # if self._strict and deref(self._chart)[index] != NULL:
        #     raise Exception("Chart already has label %s"%(key,))
        # if isinstance(chart_edges, _ChartEdge):
        #     deref(self._chart)[index] = self._hg_ptr.add_terminal_node()
        #     self._label_chart.append((index, key))

        #     self._ndata.append(1)
        #     self._nindices.append(index)
        #     self._nind.append(len(self._ndata))
        #     return



        # ind1, ind2, trans_values = self.build_up(chart_edges)

        # cdef int j = 0
        # cdef int i
        # cdef _ChartEdge chart_edge
        # for i, chart_edge in enumerate(chart_edges):
        #     edge_num = self._hg_ptr.add_edge(
        #         chart_edge.tail_ptrs)


        #     self._data += [1] * len(chart_edge.values)
        #     self._indices += chart_edge.values
        #     self._ind.append(len(self._data))


    property output_encoder:
        def __get__(self):
            return self._output_encoder

    property item_encoder:
        def __get__(self):
            return self._item_encoder

    property output_matrix:
        def __get__(self):
            return scipy.sparse.csc_matrix(
                (self._data, self._indices, self._ind),
                shape=(self._output_size,
                       self._hg_ptr.edges().size()),
                dtype=np.uint8)

    property item_matrix:
        def __get__(self):
            return scipy.sparse.csc_matrix(
                (self._ndata, self._nindices, self._nind),
                shape=(self._item_size,
                       self._hg_ptr.nodes().size()),
                dtype=np.uint8)

# {% for i in range(2, 6) %}
# cdef class IntTuple{{i}}:
#     def __cinit__(self, {% for j in range(i) %}int {{var[j]}}, {% endfor %}
#                   blank=None):
#     {% for j in range(i) %}
#         self.{{var[j]}} = {{var[j]}}
#     {% endfor %}

#     def unpack(self):
#         return ({% for j in range(i) %} self.{{var[j]}}, {% endfor %})

# cdef class IntTuple{{i}}Hasher:
#     def __cinit__(self, {% for j in range(i) %}int {{var[j]}}, {% endfor %}):
#         # n = {{i}}
#         # m = ({% for j in range(i) %} {{var[j]}},{% endfor %})
#         self._multipliers_a = 1
#         {% for j in range(1,i) %}
#         self._multipliers_{{var[j]}} = self._multipliers_{{var[j-1]}} * {{var[j-1]}}
#         {% endfor %}
#         self._max_size = {% for j in range(i) %} {{var[j]}} * {% endfor %} 1

#     def max_size(self):
#         return self._max_size

#     cpdef unhash(self, int val):
#         cdef t = []
#         cdef int v = val
#         {% for j in range(i-1, -1, -1) %}
#         #for k in range({{i}}-1, -1, -1):
#         t.insert(0, v / self._multipliers_{{var[j]}})
#         v = v % self._multipliers_{{var[j]}}
#         {% endfor %}
#         return tuple(t)



#     cpdef int hash_item(self, {% for j in range(i) %}{{var[j]}}, {% endfor %}):
#         return \
#           {% for j in range(i) %} \
#             {{var[j]}} * self._multipliers_{{var[j]}} + \
#           {% endfor %} + 0
# {% endfor %}
