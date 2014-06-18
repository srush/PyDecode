# For finite-state automaton construction.

cdef class DFA:
    def __init__(self, int num_states,
                 int num_symbols, transitions,
                 final):
        cdef vector[map[int, int]] ctransitions
        ctransitions.resize(num_states)
        for i, m in enumerate(transitions):
            for k in m:
                ctransitions[i][k] = m[k]

        cdef set[int] cfinals
        for f in final:
            cfinals.insert(f)

        self.thisptr = new CDFA(num_states, num_symbols,
                                ctransitions, cfinals)

    def is_final(self, int state):
        return self.thisptr.final(state)

    def transition(self, int state, int symbol):
        return self.thisptr.transition(state, symbol)

    def valid_transition(self, int state, int symbol):
        return self.thisptr.valid_transition(state, symbol)

cdef class DFALabel:
    cdef init(self, CDFALabel label, core):
        self.label = label
        self._core = core
        return self

    property left_state:
        def __get__(self):
            return self.label.left_state

    property right_state:
        def __get__(self):
            return self.label.right_state

    property core:
        def __get__(self):
            return self._core

    def __str__(self):
        return str(self.core) + " " + str(self.label.left_state) + " " + str(self.label.right_state)


# For lattice construction.

cdef class LatticeLabel:
    cdef init(self, CLatticeLabel label):
        self.label = label
        return self

    property i:
        def __get__(self):
            return self.label.i

    property j:
        def __get__(self):
            return self.label.j

    def __str__(self):
        return str(self.i) + " " + str(self.j)

def make_lattice(int width, int height, transitions):
    cdef vector[vector[int] ] ctrans
    cdef vector[int] tmp
    for i in range(len(transitions)):
        for j in range(len(transitions[i])):
            tmp.push_back(transitions[i][j])
        ctrans.push_back(tmp)
        tmp.clear()
    cdef Hypergraph h = Hypergraph()

    cdef vector[CLatticeLabel] clabels
    cdef CHypergraph *chyper = cmake_lattice(width, height, ctrans, &clabels)

    node_labels = [LatticeLabel().init(clabels[i])
                   for i in range(clabels.size())]
    assert(chyper.nodes().size() == len(node_labels))
    return h.init(chyper,
                  Labeling(h, node_labels, None))


def count_constrained_viterbi(Hypergraph graph,
                              _LogViterbiPotentials potentials,
                              _CountingPotentials counts,
                              int limit):
    """
    DEPRECATED
    """

    cdef CHyperpath *path = \
        ccount_constrained_viterbi(graph.thisptr,
                                   deref(potentials.thisptr),
                                   deref(counts.thisptr),
                                   limit)

    return Path().init(path, graph)


def extend_hypergraph_by_count(Hypergraph graph,
                               _CountingPotentials potentials,
                               int lower_limit,
                               int upper_limit,
                               int goal):
    """
    DEPRECATED
    """

    cdef CHypergraphMap *projection = \
        cextend_hypergraph_by_count(graph.thisptr,
                                    deref(potentials.thisptr),
                                    lower_limit,
                                    upper_limit,
                                    goal)

    return convert_hypergraph_map(projection, None, graph)


def extend_hypergraph_by_dfa(Hypergraph graph,
                             _CountingPotentials potentials,
                             DFA dfa):
    """
    DEPRECATED
    """

    cdef vector[CDFALabel] labels
    cdef CHypergraphMap *projection = \
        cextend_hypergraph_by_dfa(graph.thisptr,
                                  deref(potentials.thisptr),
                                  deref(dfa.thisptr),
                                  &labels)
    node_labels = []
    cdef const CHypernode *node
    cdef vector[const CHypernode*] new_nodes = \
        projection.domain_graph().nodes()

    for i in range(labels.size()):
        node = projection.map(new_nodes[i])
        node_labels.append(DFALabel().init(labels[i],
                                           graph.labeling.node_labels[node.id()]))

    # Build domain graph
    cdef Hypergraph range_graph = Hypergraph()
    assert(projection.domain_graph().nodes().size() == \
               len(node_labels))
    range_graph.init(projection.domain_graph(),
                     Labeling(range_graph, node_labels, None))
    return convert_hypergraph_map(projection, range_graph, graph)

def binarize(Hypergraph graph):
    """
    Binarize a hypergraph by making all k-ary edges right branching.

    Parameters
    ----------
    graph : Hypergraph

    Returns
    --------
    map : HypergraphMap
        A map from the original graph to the binary branching graph.
    """
    cdef CHypergraphMap *hypergraph_map = cbinarize(graph.thisptr)
    return convert_hypergraph_map(hypergraph_map, graph, None)
