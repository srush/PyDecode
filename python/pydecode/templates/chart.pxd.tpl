cimport numpy as np
cimport cython

cdef class IndexedEncoder:
    cdef _hasher
    cdef int _max_size
    cdef _shape
    cdef np.ndarray _multipliers
    cpdef np.ndarray transform(self, np.ndarray element)
    cpdef np.ndarray inverse_transform(self, np.ndarray index)

cdef class _ChartEdge:
    cdef vector[const CHypernode *] tail_ptrs
    cdef values
    cdef items

cdef class ChartBuilder:
    cdef _ChartEdge  _merge(self, args, outs)
    cpdef init(self, int index)
    # cpdef set(self, int key, int [:, :] chart_edges)
    cpdef set2(self, long key, long [:] edges1, long [:] edges2, long [:] labels)
    cdef CHypergraph *_hg_ptr

    cdef vector[const CHypernode *] *_chart
    cdef _label_chart

    cdef bool _done
    cdef int _last
    cdef bool _strict

    cdef _item_encoder
    cdef _item_size

    cdef _output_encoder
    cdef _output_size

    cdef int _max_arity

    cdef _data
    cdef _indices
    cdef _ind

    cdef _ndata
    cdef _nindices
    cdef _nind

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
