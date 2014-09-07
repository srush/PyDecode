import numpy as np
cimport cython

    # item_matrix : Sparse matrix |I| x |V|
    #    Maps vertices of the hypergraph to item indicators.

    # output_matrix : Sparse matrix |O| x |E|
    #    Maps hyperedges of the hypergraph to sparse output counts.


class DynamicProgram:
    def __init__(self, hypergraph,
                 # item_indices,
                 # output_indices,
                 items,
                 outputs):
        self.hypergraph = hypergraph
        self.items = items
        self.outputs = outputs
        # self.item_indices = np.array(item_indices)
        # self.output_indices = np.array(output_indices)

        self._item_matrix = None
        self._output_matrix = None

    def _make_output_matrix(self):
        # assert self._construct_output, \
        #     "Output size not specified."
        data = []
        outputs = []
        ind = [0]

        for index in self.hypergraph.labeling:
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
            ([1] * len(self.hypergraph.node_labeling),
             self.item_indices,
             range(len(self.item_indices) + 1)),
            shape=(np.max(self.items) + 1,
                   len(self.hypergraph.nodes)),
            dtype=np.uint8)

    @property
    def output_matrix(self):
        if self._output_matrix is None:
            self._output_matrix = self._make_output_matrix()
        return self._output_matrix

    @property
    def item_matrix(self):
        if self._item_matrix is None:
            self._item_matrix = self._make_item_matrix()
        return self._item_matrix

NODE_NULL = -1
cdef class ChartBuilder:
    def __dealloc__(self):
        del self._builder
        del self._chart

    def __init__(self,
                 items,
                 outputs=None,
                 unstrict=False,
                 expected_size=None,
                 lattice=False):
        self._done = False
        self._last = -1
        self._no_tail = set[long]()
        self._strict = not unstrict
        self._builder = new CHypergraphBuilder(lattice)

        self._size = np.max(items) + 1
        self.items = items
        self.outputs = outputs
        self._chart = new vector[int](self._size, NODE_NULL)

        # Output structures.
        # self._output_size = np.max(outputs) + 1
        # self._construct_output = self._output_size is not None
        self._construct_output = True
        if expected_size:
            self._builder.set_expected_size(self._size,
                                           expected_size[0],
                                           expected_size[1])
            self._max_arity = expected_size[1]

        self._out = np.array([0], dtype=np.int64)

        self._lattice = lattice

    def init(self, items):
        r"""
        init(self, items)

        Initialize a base case for a set of items.

        Formally, for each item :math:`v \in I`, sets

        .. math::

           C_v  = \bar{1}

        Parameters:
        ------------

        items : int or array
            Initialize the given items.
        """
        if isinstance(items, np.ndarray):
            return self._init_buffer(items)
        elif isinstance(items, int):
            return self._init_list([items])
        else:
            return self._init_list(items)


    def set(self, long item, tails_list, labels=None):
        r"""
        set(self, items, tails_list, labels=None)

        Add an item with a recursive definition.

        Calling this function adds a recursive definition to the dynamic program
        for ``item`` based on the previous defined values of the items in tails and
        a sequnce of labels.


        Formally, for given ``item`` :math:`v`, adds the recursive definition

        .. math::

           C_v  = \bigoplus_{i} C_{T_{i,1}} \otimes C_{T_{i,2}} \ldots \otimes C_{T_{i,n}} \otimes w(L_i)

        where :math:`T` is a sequence of tail items given by parameter ``tails``,
        :math:`L` is a sequence of labels given by parameter ``labels``.

        Warning: This function is more flexible, but often slower than ``set_t``.


        Parameters
        ----------

        item : int
            The dynamic programming item to set.

        tails_list : list of list of ints
            A list of tail lists :math:`T`.

        labels : list of ints, optional
            A list of labels. Must be None orthe same size as tails.
        """
        return self._set_list(item, tails_list, labels)


    def set_t(self, long item, tails1, tails2=None, tails3=None, labels=None):
        r"""
        set_t(self, item, tails1, tails2=None, tails3=None, labels=None)

        Add an item with a recursive definition, transpose.

        Calling this function adds a recursive definition to the dynamic program
        for ``item`` based on the previous defined values of the items in tails and
        a sequnce of labels.

        Formally, for given ``item`` :math:`v`, adds the recursive definition

        .. math::

           C_v  = \bigoplus_{i} C_{T_{1,i}} \otimes C_{T_{2,i}} \otimes C_{T_{3,i}} \otimes w(L_i)

        where :math:`T` is a sequence of tail items given by parameters ``tails1``, ``tails2``, ``tails3``
        :math:`L` is a sequence of labels given by parameter ``labels``.


        Parameters
        ----------

        item : int
            The dynamic programming item to set.

        tails1, tails2, tails3  : int ndarray, optional
            Sequence of tail items. Represented by :math:`T_1, T_2, T_3` respectively.
            Must be of the same length or None.

        labels : int ndarray, optional
            The labels for each definition. Must be the same length as tails.

        """
        return self._set_transpose(item, tails1, tails2, tails3, labels)

    def finish(self, reconstruct=False):
        r"""
        Complete the dynamic program.

        Returns
        --------
        hypergraph : :py:class:`Hypergraph`
            The hypergraph representing the dynamic program.

        """
        if self._done:
            raise Exception("Hypergraph not constructed.")
        if self._strict and self._no_tail.size() != 1:
            raise Exception("Hypergraph has multiple vertices that are not connected: %s."%(self._no_tail,))
        self._done = True
        self._hg_ptr = self._builder.finish(reconstruct)

        hypergraph = Hypergraph(self._lattice)
        hypergraph.init(self._hg_ptr,
                        Labeling(hypergraph, None))
        return hypergraph


    @cython.boundscheck(False)
    cdef _init_buffer(self, long [:] indices):
        cdef int i
        cdef long index
        for i in range(indices.shape[0]):
            index = indices[i]
            deref(self._chart)[index] = \
                self._builder.add_terminal_node(index)
            self._no_tail.insert(index)

    @cython.boundscheck(False)
    cdef _init_list(self, indices):
        cdef long index
        for index in indices:
            deref(self._chart)[index] = \
                self._builder.add_terminal_node(index)
            self._no_tail.insert(index)


    cdef _set_list(self, long index, tuples, out=None):
        deref(self._chart)[index] = self._builder.start_node(index)
        cdef vector[int] tails
        cdef int i, j, node

        blank = (out is None)
        for j, tail in enumerate(tuples):
            tails.clear()
            for node in tail:
                tails.push_back(deref(self._chart)[node])
                if tails.back() == NODE_NULL:
                    raise Exception(
                        "Item %s not found for tail."%(node,))

            if self._construct_output:
                if not blank:
                    self._builder.add_edge(tails, out[j])
                else:
                    self._builder.add_edge(tails, -1)
            else:
                self._builder.add_edge(tails, -1)
            for node in tail:
                if self._no_tail.find(node) != self._no_tail.end():
                    self._no_tail.erase(node)


        result = self._builder.end_node()
        self._finish_node(index, result)

    cdef _finish_node(self, long index, result):
        if not result:
            if self._strict:
                raise Exception("No tail items found for item %s."%(index,))
            deref(self._chart)[index] = NODE_NULL
        else:
            self._last = index
            self._no_tail.insert(index)
            # self._nindices.append(index)


    @cython.boundscheck(False)

    cdef _set_transpose(self,
                        long index,
                        long [:] tails1,
                        long [:] tails2=None,
                        long [:] tails3=None,
                        long [:] out=None):

        deref(self._chart)[index] = self._builder.start_node(index)

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
            if self._construct_output:
                if not blank:
                    self._builder.add_edge(tails, out[j])
                else:
                    self._builder.add_edge(tails, -1)
            else:
                self._builder.add_edge(tails, -1)

            if self._no_tail.find(tails1[j]) != self._no_tail.end():
                self._no_tail.erase(tails1[j])
            if tails2 is not None and self._no_tail.find(tails2[j]) != self._no_tail.end():
                self._no_tail.erase(tails2[j])
            if tails3 is not None and self._no_tail.find(tails3[j]) != self._no_tail.end():
                self._no_tail.erase(tails3[j])


            # if self._construct_output:
            #     if not blank and out[j] != -1:
            #         self._indices.append(out[j])
            #     else:
            #         self._indices.append(-1)

        result = self._builder.end_node()
        self._finish_node(index, result)
        # if not result:
        #     if self._strict:
        #         raise Exception("No tail items found for item %s."%(index,))
        #     deref(self._chart)[index] = NODE_NULL
        # else:
        #     self._last = index
        #     self._no_tail.insert(index)
        #     self._nindices.append(index)

    # DynamicProgram(hypergraph,
    #                           self.items,
    #                           self.outputs)
