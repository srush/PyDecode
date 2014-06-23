cimport numpy as np
cimport cython

# cdef class IndexedEncoder:
#     cdef _hasher
#     cdef int _max_size
#     cdef _shape
#     cdef np.ndarray _multipliers
#     cpdef np.ndarray transform(self, np.ndarray element)
#     cpdef np.ndarray inverse_transform(self, np.ndarray index)

# cdef class _ChartEdge:
#     cdef vector[const CHypernode *] tail_ptrs
#     cdef values
#     cdef items

cdef class ChartBuilder:
    cpdef init(self, long [:] index)
    cpdef set(self,
              long index,
              long [:] tails1,
              long [:] tails2=*,
              long [:] tails3=*,
              long [:] out=*)

    cdef CHypergraph *_hg_ptr
    cdef vector[const CHypernode *] *_chart

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



# {% for i in range(2, 6) %}
# cdef class IntTuple{{i}}:
#     cdef int a, b, c, d

# cdef class IntTuple{{i}}Hasher:
#     {% for j in range(i) %}
#     cdef int _multipliers_{{var[j]}}
#     {% endfor %}
#     cdef int _max_size
#     cpdef int hash_item(self, {% for j in range(i) %}int {{var[j]}}, {% endfor %})
#     cpdef unhash(self, int val)
# {% endfor %}
