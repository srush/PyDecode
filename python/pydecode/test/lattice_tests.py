import random
import pydecode
import pydecode.test.utils as utils
from collections import defaultdict
import nose.tools as nt
import numpy as np

def test_main():
    yield check_lattice
    # for hypergraph in utils.hypergraphs():
    #     log_pot = utils.random_log_viterbi_potentials_array(hypergraph)
    #     inside = utils.random_inside_potentials(hypergraph)


def check_lattice():
    width = random.randint(1, 10)
    height = random.randint(1, 10)
    lattice = pydecode.make_lattice(width, height, [range(height) for h in range(height)])
    lattice.labeling = pydecode.Labeling(lattice, [1 for node in lattice.nodes])
    for node in lattice.nodes:
        assert node.label == 1
