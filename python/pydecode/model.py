from pystruct.models import StructuredModel
from sklearn.feature_extraction import DictVectorizer
import pydecode.hyper as ph
import numpy as np
import pydecode.chart as chart


class HypergraphModelBuilder:
    """
    An interface for building a hypergraph model.

    """
    def dynamic_program(self, x, chart):
        raise NotImplementedError()

    def labels(self, x, y):
        raise NotImplementedError()

    def gen_features(self, x, label):
        raise NotImplementedError()


class HypergraphModel(StructuredModel):
    """
    An implementation of a structured model.

    """
    def __init__(self, builder):
        self._builder = builder
        self._vec = DictVectorizer()

    def _build_hypergraph(self, x):
        c = chart.ChartBuilder(lambda a: a,
                               chart.HypergraphSemiRing, True)
        return self._builder.dynamic_program(x, c).finish()

    def _build_weights(self, hypergraph, x, w):
        def weight_builder(label):
            return self._features(x, label).dot(w)
        return ph.Weights(hypergraph).build(weight_builder)

    def _features(self, x, label):
        print label
        d = self._builder.gen_features(x, label)
        return self._vec.transform({s: 1 for s in d})

    def _path_features(self, hypergraph, x, path):
        return sum([self._features(x, hypergraph.label(edge))
                    for edge in path])

    def psi(self, x, y):
        features = set()
        for label in self._builder.labels(x, y):
            features |= self._builder.gen_features(x, label)
        f2 = {s: 1 for s in features}
        return self._vec.fit_transform(f2)

    def initialize(self, X, Y):
        features = set()
        sets = (self._builder.gen_features(x, label)
                for x, y in zip(X, Y)
                for label in self._builder.labels(x, y))
        for s in sets:
            features |= s

        features2 = {s: 1 for s in features}
        t = self._vec.fit_transform(features2)

        self.size_psi = t.size

    def inference(self, x, w):
        hypergraph = self._build_hypergraph(x)
        weights = self._build_weights(hypergraph, x, w)
        path = ph.best_path(hypergraph, weights)
        return self._path_features(hypergraph, x, path)

    def loss(self, yhat, y):
        difference = 0
        for edge in yhat:
            if edge not in y:
                difference += 1
        return difference

    def max_loss(self, y):
        return sum([1 for edge in y])
