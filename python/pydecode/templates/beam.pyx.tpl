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

cdef class ParsingElement:

    cdef init(self, CParsingElement element):
        self.data = element
        return self

    def set(self, int edge, int position, int total_size):
        self.data.edge = edge
        self.data.position = position
        self.data.total_size = total_size
        self.data.recompute_hash()
        return self

    def __str__(self):
        return "E_%d_%d_%d %s"%(self.data.edge, self.data.position, self.data.total_size,
                                self.data.up != NULL)


    def to_str(self):
        return "E_%d_%d_%d"%(self.data.edge, self.data.position, self.data.total_size)

    def from_str(self, s):
        if len(s) == 0 or s[0] != "E":
            self.data.position = -1
            return self
        else:
            t = s.split("_")
            self.data.edge = int(t[1])
            self.data.position = int(t[2])
            self.data.total_size = int(t[3])
            self.data.recompute_hash()
            return self

    def inc(self):
        self.data.position += 1

    def edge(self):
        return self.data.edge
    def position(self):
        return self.data.position
    def size(self):
        return self.data.total_size

    def equal(self, ParsingElement other):
        return cparsingequal(self.data, other.data)

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
            data.append((_{{S.type}}_from_cpp(p.sig),
                         p.current_score,
                         p.future_score))
        return data


    property exact:
        def __get__(self):
            return self.thisptr.exact

cdef _{{S.type}}_from_cpp({{S.cvalue}} val):
    {% if S.from_cpp %}
    return {{S.from_cpp}}
    {% else %}
    return val
    {% endif %}

{% if S.to_cpp %}
cdef {{S.cvalue}} _{{S.type}}_to_cpp({{S.pvalue}} val):
    return <{{S.cvalue}}>{{S.to_cpp}}
{% else %}
cdef {{S.cvalue}} _{{S.type}}_to_cpp({{S.cvalue}} val):
    return val
{% endif %}

def beam_search_{{S.type}}(Hypergraph graph,
                        old_potentials,
                constraints,
                           double [:] outside,
                double lower_bound,
                groups,
                           group_limits,
                int num_groups=-1,
                bool recombine=True,
                           bool cube_pruning = False):
    r"""

    Parameters
    -----------
    graph : Hypergraph

    potentials : LogViterbiPotentials
       The potentials on each hyperedge.

    constraints : BinaryVectorPotentials
       The constraints (bitset) at each hyperedge.

    lower_bound : double

    outside : LogViterbiChart
        The outside scores.

    groups : size of vertex list
       The group for each vertex.

    group_limits :
       The size limit for each group.

    num_groups :
        The total number of groups.
    """
    cdef _LogViterbiPotentials potentials = \
                                            get_potentials(graph, old_potentials,
                                LogViterbi.Potentials)

    cdef CLogViterbiChart *out_chart = new CLogViterbiChart(
        graph.thisptr,
        &outside[0])

    if num_groups == -1:
        ngroups = max(groups) + 1
    else:
        ngroups = num_groups
    cdef vector[int] cgroups = groups
    cdef vector[int] cgroup_limits = group_limits

    cdef CBeamGroups *beam_groups = new CBeamGroups(graph.thisptr,
                                                    cgroups,
                                                    cgroup_limits,
                                                    ngroups)
    cdef vector[{{S.cvalue}}] cons
    for c in constraints:
        cons.push_back(_{{S.type}}_to_cpp(c))
    # cgroups.resize(graph.nodes_size())
    # cdef vector[int] cgroup_limits
    # cgroups.resize(graph.nodes_size())

    # for i, group in enumerate(groups):
    #     cgroups[i] = group


    cdef CBeamChart{{S.type}} *chart
    if False:
        pass
        # chart = ccube_pruning{{S.type}}(graph.thisptr,
        #              deref(potentials.thisptr),
        #              cons,
        #              deref(out_chart),
        #              lower_bound,
        #              deref(beam_groups),
        #              recombine)
    else:
        chart = cbeam_search{{S.type}}(graph.thisptr,
                     deref(potentials.thisptr),
                                       cons,
                     deref(out_chart),
                     lower_bound,
                     deref(beam_groups),
                     recombine)
    return BeamChart{{S.type}}().init(chart, graph)

{% endfor %}
