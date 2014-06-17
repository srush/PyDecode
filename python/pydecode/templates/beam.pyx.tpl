from cython.operator cimport dereference as deref
from libcpp cimport bool
from libcpp.vector cimport vector


from pydecode.potentials cimport *

cdef class Bitset:
    """
    Bitset


    """
    def __init__(self, v=-1):
        if v != -1:
            self.data[v] = 1

    cdef init(self, cbitset data):
        self.data = data
        return self

    def __getitem__(self, int position):
        return self.data[position]

    def __setitem__(self, int position, int val):
        self.data[position] = val

{% for S in semirings %}
cdef class BeamChart{{S.type}}:
    cdef init(self, CBeamChart{{S.type}} *chart, Hypergraph graph):
        self.thisptr = chart
        self.graph = graph
        return self

    def __dealloc__(self):
        if self.thisptr is not NULL:
            del self.thisptr
            self.thisptr = NULL

    def path(self, int result):
        if self.thisptr.get_path(result) == NULL:
            return None
        return Path().init(self.thisptr.get_path(result),
                           self.graph)

    def __getitem__(self, Vertex vertex):
        cdef vector[CBeamHyp{{S.type}} *] beam = \
                    self.thisptr.get_beam(vertex.nodeptr)
        data = []
        i = 0
        for p in beam:
            data.append((_{{S.from}}_from_cpp(p.sig),
                         p.current_score,
                         p.future_score))
        return data


    property exact:
        def __get__(self):
            return self.thisptr.exact


# def beam_search_{{S.from}}(Hypergraph graph,
#                 _LogViterbiPotentials potentials,
#                 {{S.type}}s constraints,
#                 outside,
#                 double lower_bound,
#                 groups,
#                 group_limits,
#                 int num_groups=-1,
#                 bool recombine=True,
#                            bool cube_pruning = False):
#     r"""

#     Parameters
#     -----------
#     graph : Hypergraph

#     potentials : LogViterbiPotentials
#        The potentials on each hyperedge.

#     constraints : BinaryVectorPotentials
#        The constraints (bitset) at each hyperedge.

#     lower_bound : double

#     outside : LogViterbiChart
#         The outside scores.

#     groups : size of vertex list
#        The group for each vertex.

#     group_limits :
#        The size limit for each group.

#     num_groups :
#         The total number of groups.
#     """
#     if num_groups == -1:
#         ngroups = max(groups) + 1
#     else:
#         ngroups = num_groups
#     cdef vector[int] cgroups = groups
#     cdef vector[int] cgroup_limits = group_limits

#     cdef CBeamGroups *beam_groups = new CBeamGroups(graph.thisptr,
#                                                     cgroups,
#                                                     cgroup_limits,
#                                                     ngroups)
#     # cgroups.resize(graph.nodes_size())
#     # cdef vector[int] cgroup_limits
#     # cgroups.resize(graph.nodes_size())

#     # for i, group in enumerate(groups):
#     #     cgroups[i] = group


#     cdef CBeamChart{{S.type}} *chart
#     if cube_pruning:
#         chart = ccube_pruning{{S.type}}(graph.thisptr,
#                      deref(potentials.thisptr),
#                      deref(constraints.thisptr),
#                      deref(outside.chart),
#                      lower_bound,
#                      deref(beam_groups),
#                      recombine)
#     else:
#         chart = cbeam_search{{S.type}}(graph.thisptr,
#                      deref(potentials.thisptr),
#                      deref(constraints.thisptr),
#                      deref(outside.chart),
#                      lower_bound,
#                      deref(beam_groups),
#                      recombine)
#     return BeamChart{{S.type}}().init(chart, graph)

{% endfor %}
