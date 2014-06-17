cdef class IndexSet:
    cdef _hasher
    cdef int _max_size
    cpdef int index(self, element)
    cpdef element(self, int index)

cdef class ChartEdge:
    cdef vector[const CHypernode *] tail_ptrs
    cdef values

cdef class ChartBuilder:
    cdef CHypergraph *_hg_ptr
    cdef vector[const CHypernode *] *_chart
    cdef _label_chart
    cdef _semiring
    cdef _scorer
    cdef bool _done
    cdef int _last
    cdef bool _debug
    cdef bool _strict
    cdef IndexSet _item_set
    cdef IndexSet _output_set
    cdef int _max_arity
    cdef _data
    cdef _indices
    cdef _ind

    cdef _ndata
    cdef _nindices
    cdef _nind

cdef class SizedTupleHasher:
    cdef vector[int] _multipliers
    cdef int _max_size
    cdef _np_multi
    cpdef int hash_item(self, t)

{% for i in range(2, 6) %}
cdef class IntTuple{{i}}:
    cdef int a, b, c, d

cdef class IntTuple{{i}}Hasher:
    {% for j in range(i) %}
    cdef int _multipliers_{{var[j]}}
    {% endfor %}
    cdef int _max_size
    cpdef int hash_item(self, {% for j in range(i) %}int {{var[j]}}, {% endfor %})
    cpdef unhash(self, int val)
{% endfor %}
