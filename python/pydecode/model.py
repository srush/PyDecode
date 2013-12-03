"""
A structured prediction and training library.
Requires pystruct.
"""

from pystruct.models import StructuredModel
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
import pydecode.hyper as ph
import numpy as np
import pydecode.chart as chart
import time
import pydecode.lp as lp

class DynamicProgrammingModel(StructuredModel):
    """
    An implementation of a structured model for dynamic programming.

    Minimally implement
    :py:method:`DynamicProgrammingModel.dynamic_program` and
    :py:method:`DynamicProgrammingModel.factored_psi` to use.

    """
    def __init__(self, constrained=False):
        self._vec = DictVectorizer()
        self._constrained = constrained

    def dynamic_program(self, x, chart):
        r"""
        Construct the dynamic program for this input example.

        Parameters
        ----------

        x : any
           The input structure

        chart : :py:class:`ChartBuilder`
           A chart builder object.
        """

        raise NotImplementedError()

    def constraints(self, x, hypergraph):
        return None
        #raise NotImplementedError()

    def factored_psi(self, x, index, data):
        """
        Compute the features for a given index.

        Parameters
        ----------

        x : any
           The input structure

        index : any
           A factored index for the problem.

        Returns
        --------

        A set of features.

        """
        raise NotImplementedError()

    def psi(self, x, y):
        features = {}
        data = self.initialize_features(x)
        for index in y:
            f = self.factored_psi(x, index, data)
            for k, v in f.iteritems():
                features.setdefault(k, 0)
                features[k] += v
                
        final_features = self._vec.transform(features)
        #print "Features:", self._vec.inverse_transform(final_features)
        return final_features

    def initialize(self, X, Y):
        features = {}
        sets = []
        for x, y in zip(X, Y):
            data = self.initialize_features(x)
            sets += [self.factored_psi(x, index, data)
                     for index in y]
        for s in sets:
            for k, v in s.iteritems():
                features.setdefault(k, 0)
                features[k] += v

        t = self._vec.fit_transform(features)
        self.size_psi = t.size

    def inference(self, x, w, relaxed=False):


        hypergraph = self._build_hypergraph(x)
        potentials = self._build_potentials(hypergraph, x, w)
        if not self._constrained:
            path = ph.best_path(hypergraph, potentials)
        else:
            constraints = self.constraints(x, hypergraph)
            hyperlp = lp.HypergraphLP.make_lp(hypergraph, potentials, integral=True)
            hyperlp.add_constraints(constraints)
            hyperlp.pulp.solvers.GLPK(mip=1)
            path = hyperlp.path
        y = set()
        for edge in path:
            y.add(hypergraph.label(edge))
        self.psi(x, y)
        return y

            #print repr(hypergraph.label(edge))
        # print len(path.edges)
        #print "DONE"

        # a = time.time()
        # print time.time() - a
        # b = time.time()
        # print time.time() - b
        #print "SCORE IS:", potentials.dot(path)

    def loss(self, yhat, y):
        difference = 0
        # print "GOLD", y
        # print "CHECK", yhat
        ydiff = set()
        for y1 in y:
            if y1 not in yhat:
                difference += 1
                ydiff.add(y1) 
        #print "DIFF", difference, ydiff
        return difference

    def max_loss(self, y):
        return sum([1 for index in y])

    def _build_hypergraph(self, x):
        c = chart.ChartBuilder(lambda a: a,
                               chart.HypergraphSemiRing, True)
        self.dynamic_program(x, c)
        return c.finish()

    def _build_potentials(self, hypergraph, x, w):
        data = self.initialize_features(x)
        features = [self.factored_psi(x, hypergraph.label(edge), data) for edge in hypergraph.edges]
        f = self._vec.transform(features)
        scores =  f * w.T
        #print "Weights:", self._vec.inverse_transform(w) 
        #print 
        return ph.Potentials(hypergraph).from_vector(scores)

        #return weights
        #print len(), len(hypergraph.edges)
        # def potential_builder(index):
        #     return 
        # return ph.Potentials(hypergraph).build(potential_builder)
        # def potential_builder(index):
        #      return self._features(x, index, data).dot(w.T)
        # w2 = ph.Potentials(hypergraph).build(potential_builder)
        # for edge in hypergraph.edges:
        #     assert weights[edge] == w2[edge]


    # def _features(self, x, index, data):
    #     d = self.factored_psi(x, index, data)
    #     # This is slow. 
    #     return self._vec.transform(d) # {s: 1 for s in d})

    # def _path_features(self, hypergraph, x, path, data):
    #     return sum((self._features(x, hypergraph.label(edge), data)
    #                 for edge in path))


# hm = TaggingCRFModel()
# hm.initialize(data_X, data_Y)
# for i in range(len(data_X))[:10]:
#     s = set(data_Y[i])
#     c = chart.ChartBuilder(lambda a: a,
#                            chart.HypergraphSemiRing, True)
#     hm.dynamic_program(data_X[i], c)
#     h = c.finish()
#     bool_pot = ph.BoolPotentials(h).build(lambda a: a in s)
#     path = ph.best_path(h, bool_pot)
#     #for edge in path: print h.label(edge)
#     assert bool_pot.dot(path)
