import numpy as np
cimport cython

    # item_matrix : Sparse matrix |I| x |V|
    #    Maps vertices of the hypergraph to item indicators.

    # output_matrix : Sparse matrix |O| x |E|
    #    Maps hyperedges of the hypergraph to sparse output counts.


cdef class DynamicProgram:
    def __init__(self, hypergraph,
                 item_indices,
                 output_indices,
                 items, outputs):
        self._hypergraph = hypergraph
        self._items = items
        self._output_shape = outputs.shape
        self._outputs = outputs
        self._item_indices = np.array(item_indices)
        self._output_indices = np.array(output_indices)

        self._item_matrix = None
        self._output_matrix = None
        self._active_outputs = None
        self._active_output_indices = None

    def _make_output_matrix(self):
        # assert self._construct_output, \
        #     "Output size not specified."
        data = []
        outputs = []
        ind = [0]
        for index in self.output_indices:
            if index != -1:
                data.append(1)
                outputs.append(index)
            ind.append(len(data))
        return scipy.sparse.csc_matrix(
            (data, outputs, ind),
            shape=(np.max(self.outputs) + 1,
                   len(self.hypergraph.edges)),
            dtype=np.uint8)

    def _make_item_matrix(self):
        return scipy.sparse.csc_matrix(
            ([1] * len(self.item_indices),
             self.item_indices,
             range(len(self.item_indices) + 1)),
            shape=(np.max(self.items) + 1,
                   len(self.hypergraph.nodes)),
            dtype=np.uint8)

    def _make_active_outputs(self):
        d = {}
        counter = 0
        aoi = []
        ao = []
        cdef int index
        for index in self._output_indices:
            if index == -1:
                aoi.append(-1)
            elif index in d:
                aoi.append(d[index])
            else:
                aoi.append(counter)
                ao.append(index)
                d[index] = counter
                counter += 1
        self._active_output_indices = np.array(aoi)
        self._active_outputs = np.array(ao)

    property hypergraph:
        def __get__(self):
            return self._hypergraph

    property items:
        def __get__(self):
            return self._items

    property outputs:
        def __get__(self):
            return self._outputs

    property output_shape:
        def __get__(self):
            return self._output_shape

    property item_indices:
        def __get__(self):
            return self._item_indices

    property output_indices:
        def __get__(self):
            return self._output_indices

    property output_matrix:
        def __get__(self):
            if self._output_matrix is None:
                self._output_matrix = self._make_output_matrix()
            return self._output_matrix

    property item_matrix:
        def __get__(self):
            if self._item_matrix is None:
                self._item_matrix = self._make_item_matrix()
            return self._item_matrix

    property active_outputs:
        def __get__(self):
            if self._active_outputs is None:
                self._make_active_outputs()
            return self._active_outputs

    property active_output_indices:
        def __get__(self):
            if self._active_output_indices is None:
                self._make_active_outputs()
            return self._active_output_indices

cdef class _ChartEdge:
    pass

NODE_NULL = -1
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
    items : Encoder I -> {0...|I|}
       Encodes elements of the item set I as integers.

    outputs : Encoder O -> {0...|O|}
       Encodes elements of the output set O as integers.
    """

    def __init__(self,
                 items,
                 outputs,
                 unstrict=False,
                 expected_size=None,
                 lattice=False):
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
        self._hg_ptr = new CHypergraph(lattice)

        self._size = np.max(items) + 1
        self.items = items
        self.outputs = outputs
        self._chart = new vector[int](self._size, NODE_NULL)
        self._nindices = []

        # Output structures.
        self._output_size = np.max(outputs) + 1
        self._construct_output = self._output_size is not None

        self._indices = []

        if expected_size:
            self._hg_ptr.set_expected_size(self._size,
                                           expected_size[0],
                                           expected_size[1])
            self._max_arity = expected_size[1]

        self._out = np.array([0], dtype=np.int64)

        self._lattice = lattice

    @cython.boundscheck(False)
    cpdef init(self, long [:] indices):
        cdef int i
        for i in range(indices.shape[0]):
            deref(self._chart)[indices[i]] = \
                self._hg_ptr.add_terminal_node()

            self._nindices.append(indices[i])
            self._no_tail.insert(indices[i])

    @cython.boundscheck(False)
    cpdef set(self,
              long index,
              long [:] tails1,
              long [:] tails2=None,
              long [:] tails3=None,
              long [:] out=None):

        deref(self._chart)[index] = self._hg_ptr.start_node()

        blank = (out is None)
        cdef vector[int] tails
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

            if tails1[j] == NODE_NULL: continue
            tails.push_back(deref(self._chart)[tails1[j]])
            if tails.back() == NODE_NULL:
                raise Exception(
                    "Item %s not found for tail 1."%(tails1[j],))

            if tails2 is not None:
                if tails2[j] == NODE_NULL: continue
                tails.push_back(deref(self._chart)[tails2[j]])

                if tails.back() == NODE_NULL:
                    raise Exception(
                        "Item %s not found for tail 2."%(tails2[j],))

            if tails3 is not None:
                if tails3[j] == NODE_NULL: continue
                tails.push_back(deref(self._chart)[tails3[j]])
                if tails.back() == NODE_NULL:
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
                else:
                    self._indices.append(-1)

        result = self._hg_ptr.end_node()
        if not result:
            if self._strict:
                raise Exception("No tail items found for item %s."%(index,))
            deref(self._chart)[index] = NODE_NULL
        else:
            self._last = index
            self._no_tail.insert(index)
            self._nindices.append(index)

    def finish(self, reconstruct=False):
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

        hypergraph = Hypergraph(self._lattice)
        hypergraph.init(self._hg_ptr,
                        Labeling(hypergraph, None))
        return DynamicProgram(hypergraph,
                              self._nindices,
                              self._indices,
                              self.items,
                              self.outputs)
