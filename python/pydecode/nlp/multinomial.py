import numpy as np
import pydecode.hyper as ph
from collections import defaultdict
import math

class Multinomial:
    def __init__(self):
        self.distribution = None

    def estimate(self, counts):
        self.distribution = counts / np.sum(counts)

class MultinomialTable:
    def __init__(self):
        self.counts = defaultdict(lambda:defaultdict(lambda:0))
        self.probs = {}

    def reestimate(self):
        for key, counts in self.counts.iteritems():
            norm = sum(counts.itervalues())
            self.probs[key] = {k: v / norm for k, v in counts.iteritems() }
        self.counts = defaultdict(lambda:defaultdict(lambda:0))

    def spike(self):
        for key, probs in self.probs.iteritems():
            print probs.items()
            best, _ = max(probs.items(), key=lambda a: a[1])
            self.probs[key] = defaultdict(lambda:0.0)
            self.probs[key][best] = 1.0



    def inc(self, position, val):
        self.counts[position[0]][position[1]] += val

    def to_array(self, hypergraph, label_map):
        arr = np.zeros([len(hypergraph.edges)])
        for i, edge in enumerate(hypergraph.edges):
            pos = label_map(edge.head.label)
            arr[i] = self.probs[pos[0]][pos[1]]
        return arr

    def show(self):
        for outer, dist in self.probs.iteritems():
            for inner, val in dist.iteritems():
                print outer, inner, val
            print



def em(distribution_table, label_map, hypergraph, base=None,
       epochs=10):
    base_potentials = base
    if base is None:
        base_potentials = np.zeros([len(hypergraph)])

    ll = []
    for i in range(epochs):
        print "epoch:", i
        potentials = ph.LogProbPotentials(hypergraph).from_array(
            base_potentials + np.log(distribution_table.to_array(hypergraph, label_map)))

        print "start"
        margs = ph.compute_marginals(hypergraph, potentials)
        print "stop"
        for node in hypergraph.nodes:
            distribution_table.inc(
                label_map(node.label),
                math.exp(margs[node] - margs[hypergraph.root]))
        distribution_table.reestimate()
        print margs[hypergraph.root]
        ll.append(margs[hypergraph.root])
    return ll
