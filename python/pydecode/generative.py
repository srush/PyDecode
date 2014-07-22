
import numpy as np
import numpy.random

def max_likelihood(array):
    array /= np.sum(array)

def addone(array):
    array += 1
    array /= np.sum(array)

def addhalf(array):
    array += 0.5
    array /= np.sum(array)

class GenerativeEstimator:
    def __init__(self, model, estimator):
        self._model = model
        self._estimator = estimator
        self.w = None
        self.exp_w = None

    def initialize(self):
        self._templates = self._model._feature._feature_templates
        self._struct = self._model._structured_features.structure()
        self._starts = np.append(self._model._feature._starts,
                                 self._model.size_joint_feature)
        self._features = self._model._structured_features


    def _distributions(self):
        for i, template in enumerate(self._templates):
            start = self._starts[i]
            end = self._starts[i+1]
            t = self.exp_w[start:end]
            t = t.reshape(template)
            cond = tuple(template[:self._struct[i]])
            yield t, cond

    def estimate(self, stats):
        self.exp_w = stats
        for t, cond in self._distributions():
            for s in np.ndindex(*cond):
                self._estimator(t[s])
        self.w = np.log(self.exp_w)
        # self.w = np.log(self.w)

    def fit(self, X, Y):
        self._model.initialize(X, Y)
        self.initialize()
        stats = np.zeros(self._model.size_joint_feature, dtype=np.float64)
        for x, y in zip(X, Y):
            stats += self._model.joint_feature(x, y)
        self.estimate(stats)


    def fit_em(self, X, Y, epochs=10):
        self._model.initialize(X, Y)
        self.initialize()
        stats = numpy.random.random(self._model.size_joint_feature)
        self.estimate(stats)

        for epoch in range(epochs):
            log_likelihood = 0.0
            stats = np.zeros(self._model.size_joint_feature, dtype=np.float64)
            for x, _ in zip(X, Y):
                log_likelihood += self._model.expected_counts(x, self.exp_w, stats)
            print epoch, log_likelihood
            self.estimate(stats)

    def show(self):
        for i, (t, cond) in enumerate(self._distributions()):
            print "Distribution", i
            for s in np.ndindex(*cond):
                print "\t","Conditioned on:", self._features.pretty_cond(i, s)
                print "\t", t[s]
