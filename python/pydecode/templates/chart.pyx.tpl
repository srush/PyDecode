import numpy as np
cimport cython

# class IdHasher:
#     def hash_item(self, i):
#         return i

# cdef class IndexedEncoder:
#     """
#     Encoder mapping integer tuples to integers.

#     Encodes the mapping N_1 x N_2 x ... N_k -> {0...|N_1||N_2|...|N_k|}.
#     Attributes
#     ----------
#     max_size : int
#         The range size |N_1||N_2|...|N_k|.

#     """
#     def __cinit__(self, shape):
#         """
#         Initialize the encoder from set sizes.

#         Parameters
#         -----------
#         size : int tuple
#            A tuple of (|N_1|, |N_2|, ..., |N_k|)

#         """
#         self._multipliers = np.zeros([len(shape), 1], dtype=np.int32)
#         self._shape = shape
#         self._multipliers[0] = 1

#         for j in range(1, len(shape)):
#             self._multipliers[j] = self._multipliers[j-1] * shape[j-1]
#         self._max_size = np.product(shape)

#     cpdef np.ndarray transform(self, np.ndarray element):
#         """
#         Transform from tuple to int.
#         """
#         return np.dot(self._multipliers.T, element.T)

#     cpdef np.ndarray inverse_transform(self, np.ndarray indices):
#         """
#         Inverse transform from int to tuple.
#         """
#         v = indices
#         m = np.zeros((len(self._multipliers), len(indices)), dtype=np.int32)
#         for j in range(len(self._multipliers) - 1, -1, -1):
#             m[j, :] =  v // self._multipliers[j]
#             v = v % self._multipliers[j]
#         return m

#     def reshape(self, matrix):
#         assert matrix.shape == self._shape
#         return np.reshape(matrix.T, (self.max_size, 1))


#     def item_vector(self, elements):
#         data = []
#         indices = []
#         ind = [0]
#         for element in elements:
#             data.append(1)
#             indices.append(self.transform(element))
#         ind.append(len(data))
#         return scipy.sparse.csc_matrix(
#             (data, indices, ind),
#             shape=(self._max_size, 1),
#             dtype=np.uint8)

#     property max_size:
#         def __get__(self):
#             return self._max_size

#         self._multipliers[0] = 1

#         for j in range(1, len(size)):
#             self._multipliers[j] = self._multiplier[j-1] * size[j-1]


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


class DynamicProgram:
    def __init__(self, hypergraph,
                 item_matrix, output_matrix,
                 items, outputs):
        self.hypergraph = hypergraph
        self.item_matrix = item_matrix
        self.output_matrix = output_matrix
        self.items = items
        self.outputs = outputs

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
                 items,
                 outputs,
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
        self._no_tail = set[long]()
        self._strict = not unstrict
        self._hg_ptr = new CHypergraph(False)

        self._size = np.max(items) + 1
        self.items = items
        self.outputs = outputs
        self._chart = new vector[const CHypernode *](self._size, NULL)

        self._ndata = []
        self._nindices = []
        self._nind = [0]

        # Output structures.
        self._output_size = np.max(outputs) + 1
        self._construct_output = self._output_size is not None

        self._data = []
        self._indices = []
        self._ind = [0]

        if expected_size:
            self._hg_ptr.set_expected_size(self._size,
                                           expected_size[0],
                                           expected_size[1])
            self._max_arity = expected_size[1]

        self._edges1 = np.array([0], dtype=np.int64)
        self._edges2 = np.array([0], dtype=np.int64)
        self._out = np.array([0], dtype=np.int64)



    @cython.boundscheck(False)
    cpdef init(self, long [:] indices):
        cdef int i
        for i in range(indices.shape[0]):
            deref(self._chart)[indices[i]] = \
                self._hg_ptr.add_terminal_node()

            self._ndata.append(1)
            self._nindices.append(indices[i])
            self._nind.append(len(self._ndata))
            self._no_tail.insert(indices[i])

    # def set(self, long index, edges1, edges2=None, out=None):
    #     if isinstance(edges1, int):
    #         self._edges1[0] = edges1
    #         self._edges2[0] = edges2
    #         self._out[0] = out
    #         return self.set2(index, self._edges1, self._edges2,
    #                          self._out[0])
    #     else:
    #         return self.set2(index, edges1, edges2, out)


    @cython.boundscheck(False)
    cpdef set(self,
              long index,
              long [:] tails1,
              long [:] tails2=None,
              long [:] tails3=None,
              long [:] out=None):

        deref(self._chart)[index] = self._hg_ptr.start_node()

        blank = (out is None)
        cdef vector[const CHypernode *] tails
        if tails2 is not None:
            assert tails1.shape[0] == tails2.shape[0], \
                "Tails 1 shape: %s Tails 2 shape: %s"% (tails1.shape[0], tails2.shape[0])

        if tails3 is not None:
            assert tails1.shape[0] == tails3.shape[0], \
                "Tails 1 shape: %s Tails 3 shape: %s"% (tails1.shape[0], tails3.shape[0])

        if out is not None:
            assert tails1.shape[0] == out.shape[0], \
                "Tails 1 shape: %s Out shape: %s"% (tails1.shape[0], out.shape[0])

        cdef int i, j

        # assert blank_edge or tails1.shape[0] == tails2.shape[0]
        for j in range(tails1.shape[0]):
            tails.clear()
            if tails1[j] == -1: continue
            tails.push_back(deref(self._chart)[tails1[j]])
            if tails.back() == NULL:
                raise Exception(
                    "Item %s not found for tail 1."%(tails1[j],))

            if tails2 is not None:
                if tails2[j] == -1: continue
                tails.push_back(deref(self._chart)[tails2[j]])

                if tails.back() == NULL:
                    raise Exception(
                        "Item %s not found for tail 2."%(tails2[j],))

            if tails3 is not None:
                if tails3[j] == -1: continue
                tails.push_back(deref(self._chart)[tails3[j]])
                if tails.back() == NULL:
                    raise Exception(
                        "Item %s not found for tail 3."%(tails3[j],))

            self._hg_ptr.add_edge(tails)

            if self._no_tail.find(tails1[j]) != self._no_tail.end():
                self._no_tail.erase(tails1[j])
            if tails2 is not None and self._no_tail.find(tails2[j]) != self._no_tail.end():
                self._no_tail.erase(tails2[j])
            if tails3 is not None and self._no_tail.find(tails3[j]) != self._no_tail.end():
                self._no_tail.erase(tails3[j])

            if self._construct_output:
                if not blank and out[j] != -1:
                    self._indices.append(out[j])
                    self._data.append(1)
                self._ind.append(len(self._data))

        result = self._hg_ptr.end_node()
        if not result:
            if self._strict:
                raise Exception("No tail items found for item %s."%(index,))
            deref(self._chart)[index] = NULL
        else:
            self._last = index
            self._no_tail.insert(index)
            self._ndata.append(1)
            self._nindices.append(index)
            self._nind.append(len(self._ndata))

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
        if self._no_tail.size() != 1:
            raise Exception("Hypergraph has multiple vertices that are not connected: %s."%(self._no_tail,))
        self._done = True
        self._hg_ptr.finish(reconstruct)

        hypergraph = Hypergraph(False)
        hypergraph.init(self._hg_ptr,
                        Labeling(hypergraph, None))
        return DynamicProgram(hypergraph,
                              self._make_item_matrix(),
                              self._make_output_matrix(),
                              self.items,
                              self.outputs)


    def _make_output_matrix(self):
        assert self._construct_output, \
            "Output size not specified."
        return scipy.sparse.csc_matrix(
            (self._data, self._indices, self._ind),
            shape=(self._output_size,
                   self._hg_ptr.edges().size()),
            dtype=np.uint8)

    def _make_item_matrix(self):
        return scipy.sparse.csc_matrix(
            (self._ndata, self._nindices, self._nind),
            shape=(self._size,
                   self._hg_ptr.nodes().size()),
            dtype=np.uint8)
