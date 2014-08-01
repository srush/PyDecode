cimport numpy as np
cimport cython

cdef class ChartBuilder:
    cpdef init(self, long [:] index)
    cpdef set(self,
              long index,
              long [:] tails1,
              long [:] tails2=*,
              long [:] tails3=*,
              long [:] labels=*,
              long [:] out=*)
    cpdef set_list(self, long index, tuples, out=*)
    cdef _finish_node(self, long index, result)

    cdef CHypergraph *_hg_ptr
    cdef CHypergraphBuilder *_builder
    cdef vector[int] *_chart

    cdef bool _done
    cdef int _last
    cdef set[long] _no_tail
    cdef bool _strict
    cdef int _max_arity

    cdef int _size
    cdef items
    cdef outputs

    cdef _ndata
    cdef _nindices
    cdef _nind


    cdef int _output_size
    cdef bool _construct_output
    cdef _data
    cdef _indices
    cdef _ind


    cdef np.ndarray _edges1
    cdef np.ndarray _edges2
    cdef np.ndarray _out

    cdef _lattice
