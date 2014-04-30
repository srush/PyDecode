from cython.operator cimport dereference as deref
from libcpp cimport bool
from libcpp.vector cimport vector


from pydecode.potentials cimport *

cdef class Bitset:
    """
    Bitset


    """

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

    def path(self, int result):
        return Path().init(self.thisptr.get_path(result),
                           self.graph)

    def __getitem__(self, Vertex vertex):
        cdef vector[CBeamHyp{{S.type}} *]             self.thisptr.get_beam(vertex.nodeptr)
        data = []
        i = 0
        for p in beam:
            data.append((_{{S.from}}_from_cpp(p.sig),
                         p.current_score,
                         p.future_score))
        return data


def beam_search_{{S.from}}(Hypergraph graph,
                LogViterbiPotentials potentials,
                {{S.type}}s constraints,
                LogViterbiChart outside,
                double lower_bound,
                groups,
                group_limits,
                int num_groups):
    r"""

    Parameters
    -----------
    graph : Hypergraph

    potentials : LogViterbiPotentials
       The potentials on each hyperedge.

    constraints : BinaryVectorPotentials
       The constraints (bitset) at each hyperedge.

    lower_bound : double

    groups : size of vetex list
       The group for each vertex.

    group_limits :
       The size limit for each group.

    num_groups :
        The total number of groups.
    """

    cdef vector[int] cgroups = groups
    cdef vector[int] cgroup_limits = group_limits
    cdef CBeamGroups *beam_groups = new CBeamGroups(graph.thisptr,
                                                    cgroups,
                                                    cgroup_limits,
                                                    num_groups)
    # cgroups.resize(graph.nodes_size())
    # cdef vector[int] cgroup_limits
    # cgroups.resize(graph.nodes_size())

    # for i, group in enumerate(groups):
    #     cgroups[i] = group


    cdef CBeamChart{{S.type}} *chart = \
        cbeam_search{{S.type}}(graph.thisptr,
                     deref(potentials.thisptr),
                     deref(constraints.thisptr),
                     deref(outside.chart),
                     lower_bound,
                     deref(beam_groups))
    return BeamChart{{S.type}}().init(chart, graph)

{% endfor %}
