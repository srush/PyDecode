"""
Structured Prediction library.
"""

from pystruct.models import StructuredModel
from sklearn.feature_extraction import DictVectorizer
import pydecode.hyper as ph
import numpy as np
import pydecode.chart as chart


class DynamicProgrammingModel(StructuredModel):
    """
    An implementation of a structured model for dynamic programming.

    Minimally implement
    :py:method:`DynamicProgrammingModel.dynamic_program` and
    :py:method:`DynamicProgrammingModel.factored_psi` to use.

    """
    def __init__(self):
        self._vec = DictVectorizer()

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

    def factored_psi(self, x, index):
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
        features = set()
        for index in y:
            features |= self.factored_psi(x, index)
        f2 = {s: 1 for s in features}
        final_features = self._vec.transform(f2)
        return final_features

    def initialize(self, X, Y):
        features = set()
        sets = (self.factored_psi(x, index)
                for x, y in zip(X, Y)
                for index in y)
        for s in sets:
            features |= s

        features2 = {s: 1 for s in features}
        t = self._vec.fit_transform(features2)
        self.size_psi = t.size

    def inference(self, x, w):
        hypergraph = self._build_hypergraph(x)
        potentials = self._build_potentials(hypergraph, x, w)
        path = ph.best_path(hypergraph, potentials)
        y = set()
        for edge in path:
            y.add(hypergraph.label(edge))
        return y

    def loss(self, yhat, y):
        difference = 0
        for y1 in y:
            if y1 not in yhat:
                difference += 1
        return difference

    def max_loss(self, y):
        return sum([1 for index in y])

    def _build_hypergraph(self, x):
        c = chart.ChartBuilder(lambda a: a,
                               chart.HypergraphSemiRing, True)
        self.dynamic_program(x, c)
        return c.finish()

    def _build_potentials(self, hypergraph, x, w):
        def potential_builder(index):
            return self._features(x, index).dot(w.T)
        return ph.Potentials(hypergraph).build(potential_builder)

    def _features(self, x, index):
        d = self.factored_psi(x, index)
        return self._vec.transform({s: 1 for s in d})

    def _path_features(self, hypergraph, x, path):
        return sum([self._features(x, hypergraph.label(edge))
                    for edge in path])
