"""
Implements CCM-like model using dependency parsing.

Dan Klein and Chris Manning
A Generative Constituent-Context Model for Improved Grammar Induction, 2002
"""
import pydecode.nlp.dependency_parsing as dep
import numpy as np
from pydecode.nlp.dependency_parsing_train import *
from pydecode.generative import *
import pydecode.model

class DepProbs(object):
    ENC = ParsingPreprocessor
    def __init__(self):
        self._prep = self.ENC()
        self.components = [self._prep]

    def feature_templates(self, _):
        def size(t):
            return self._prep.size(t) + 1
        return [(size(self.ENC.TAG), 2, 2, size(self.ENC.TAG))]

    def output_features(self, x, outputs):
        p = self._prep.preprocess(x)
        o = outputs
        c = dep.FirstOrderStopCoder
        # p(mod | head, dir)
        mod = p[self.ENC.TAG][o[:, c.MOD]]
        mod[o[:, 3]] = 0

        t = [(p[self.ENC.TAG][o[:, c.HEAD]],
              o[:, c.MOD] < o[:, c.HEAD],
              o[:, 2],
              mod)]
        return t, [0]

    def structure(self):
        return [3]

    def pretty_cond(self, template, val):
        return self._prep.inverse_transform_label(1, val[0]), val[1], val[2]

class FirstOrderGenModel(pydecode.model.HammingLossModel,
                         pydecode.model.DynamicProgrammingModel):

    def __init__(self, features, caching=True):
        super(FirstOrderGenModel, self).__init__(
            output_encoder=dep.FirstOrderStopCoder(),
            structured_features=features,
            caching=caching)
        self._decoder = dep.FirstOrderStopDecoder(use_cache=True)

    def dynamic_program(self, x):
        problem = dep.DependencyProblem(len(x.words)-1)
        return self._decoder.dynamic_program(problem)

def main():

    # coder = dep.FirstOrderStopCoder()
    # print coder.transform(dep.DependencyParse([-1, 3, 3, 0]))
    # exit()

    X, Y = read_parse("notebooks/data/wsj_sec_2_21_gold_dependencies", 40000, 15)


    features = DepProbs()
    model = FirstOrderGenModel(features, caching=True)
    ge = GenerativeEstimator(model, max_likelihood)
    ge.fit_em(X, Y, epochs=100)
    # ge.show()
    # w = estimate(model, X[:2000], Y[:2000], max_likelihood)

    score(model, ge.w, X[500:600], Y[500:600])

if __name__ == "__main__":
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        main()
