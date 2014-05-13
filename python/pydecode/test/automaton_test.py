import pydecode.hyper as ph
import pydecode.test.utils as utils
from collections import defaultdict
import nose.tools as nt
import numpy as np
import random

def lat(label):
    return str(label.i) + " " + str(label.j)
def test_main():

    hypergraph = ph.make_lattice(5, 3, [[0, 1 ,2]] * 3)
    dfa =ph.DFA(4, 4, [{0:0, 1:1, 2:2} , {0:1, 1:1, 3:3}, {0:2, 2: 2, 3:3}], [3])
    vec = [(edge.head.label.j if (edge.head.label.i not in [0, 6]) else 0)
                      for edge in hypergraph.edges]

    vec[-1] = 3
    vec[-2] = 3
    vec[-3] = 3
    counts = ph.CountingPotentials(hypergraph).from_vector(vec)
    hmap = ph.extend_hypergraph_by_dfa(hypergraph, counts, dfa)
    for n in hmap.domain_hypergraph.nodes:
        print lat(n.label.core), n.label.left_state, n.label.right_state
    path = utils.random_path(hmap.domain_hypergraph)
    print [lat(edge.head.label.core) for edge in path.edges]
    print [(edge.head.label.core.j, counts[hmap[edge]]) for edge in path.edges]
    print [(edge.head.label.left_state, edge.head.label.right_state) for edge in path.edges]

if __name__ == "__main__":
    test_main()
