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
import pulp
import sys


class DynamicProgrammingModel(StructuredModel):
    """
    An implementation of a structured model for dynamic programming.

    Minimally implement
    :py:method:`DynamicProgrammingModel.dynamic_program` and
    :py:method:`DynamicProgrammingModel.factored_joint_feature` to use.

    """
    def __init__(self, constrained=False, use_gurobi=True,
                 use_relaxed=False):
        self._vec = DictVectorizer()
        self._constrained = constrained
        self._use_gurobi = use_gurobi
        self._use_relaxed = use_relaxed
        self._debug = False
        self.inference_calls = 0

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

    def factored_joint_feature(self, x, index, data):
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

    def joint_feature(self, x, y, fractions=None):
        if not fractions:
            fractions = [1] * len(y)
        assert(len(fractions) == len(y))
        features = {}
        data = self.initialize_features(x)
        for index, frac in zip(y, fractions):
            f = self.factored_joint_feature(x, index, data)
            for k, v in f.iteritems():
                features.setdefault(k, 0)
                features[k] += v * frac

        final_features = self._vec.transform(features)
        return np.array(final_features.todense()).flatten()

    def initialize(self, X, Y):
        features = {}
        sets = []
        for x, y in zip(X, Y):
            data = self.initialize_features(x)
            sets += [self.factored_joint_feature(x, index, data)
                     for index in y]
        for s in sets:
            for k, v in s.iteritems():
                features.setdefault(k, 0)
                features[k] += v

        t = self._vec.fit_transform(features)
        self.size_joint_feature = t.size
        self.size_psi = t.size

    def inference(self, x, w, relaxed=False):
        self.inference_calls += 1
        relaxed = relaxed or self._use_relaxed
        if self._debug:
            a = time.time()

        # Build the hypergraph.
        hypergraph = self._build_hypergraph(x)
        if self._debug:
            print >>sys.stderr, "BUILD HYPERGRAPH:", time.time() - a


        # Build the potentials.
        if self._debug:
            a = time.time()
        potentials = self._build_potentials(hypergraph, x, w)
        if self._debug:
            print >>sys.stderr, "BUILD POTENTIALS:", time.time() - a


        if not self._constrained:
            # Call best path.
            if self._debug:
                a = time.time()
            path = ph.best_path(hypergraph, potentials)
            if self._debug:
                print >>sys.stderr, "BEST PATH:", time.time() - a
        else:
            if self._debug:
                a = time.time()
            constraints = self.constraints(x, hypergraph)
            hyperlp = lp.HypergraphLP.make_lp(hypergraph,
                                              potentials,
                                              integral=not relaxed)
            hyperlp.add_constraints(constraints)
            if self._debug:
                print >>sys.stderr, "BUILD LP:", time.time() - a

            if self._debug:
                a = time.time()
            if self._use_gurobi:
                hyperlp.solve(pulp.solvers.GUROBI(mip=1 if not relaxed else 0))
            else:
                hyperlp.solve(pulp.solvers.GLPK(mip=1 if not relaxed else 0))
            if self._debug:
                print >>sys.stderr, "SOLVE LP:", time.time() - a

            if relaxed:
                path = hyperlp.decode_fractional()
            else:
                path = hyperlp.path
        if self._debug:
            print

        # Returns y.
        y = set([edge.label for edge in path])
        return y

    def loss(self, yhat, y):
        difference = 0
        ydiff = set()
        for y1 in y:
            if y1 not in yhat:
                difference += 1
                ydiff.add(y1)
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
        features = [self.factored_joint_feature(x, edge.label, data)
                    for edge in hypergraph.edges]
        f = self._vec.transform(features)
        scores = f * w.T
        return ph.LogViterbiPotentials(hypergraph).from_vector(scores)


