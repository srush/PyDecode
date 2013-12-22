from cython.operator cimport dereference as deref
from libcpp cimport bool
from libcpp.vector cimport vector

from pydecode.potentials cimport *

cdef class BeamChart:

    cdef init(self, CBeamChart *chart, Hypergraph graph):
        self.thisptr = chart
        self.graph = graph
        return self

    def path(self, int result):
        return Path().init(self.thisptr.get_path(result),
                           self.graph)

    def __getitem__(self, Node node):
        cdef vector[CBeamHyp *] beam = self.thisptr.get_beam(node.nodeptr)
        data = []
        i = 0
        for p in beam:
            data.append((Bitset().init(p.sig),
                         p.current_score,
                         p.future_score))
        return data

def beam_search(Hypergraph graph,
                LogViterbiPotentials potentials,
                BinaryVectorPotentials constraints,
                LogViterbiChart outside,
                double lower_bound,
                groups,
                group_limits,
                int num_groups):
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

    cdef CBeamChart *chart = \
        cbeam_search(graph.thisptr,
                     deref(potentials.thisptr),
                     deref(constraints.thisptr),
                     deref(outside.chart),
                     lower_bound,
                     deref(beam_groups)
)
    return BeamChart().init(chart, graph)
