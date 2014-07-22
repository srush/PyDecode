import pydecode
import pydecode.model
import pydecode.nlp.dependency_parsing as dep
import numpy as np
import pydecode.nlp.format as nlpformat
import pydecode.preprocess
import pydecode.features
from collections import defaultdict, Counter, namedtuple
from pystruct.learners import StructuredPerceptron

class FirstOrderModel(pydecode.model.HammingLossModel,
                      pydecode.model.DynamicProgrammingModel):

    def __init__(self, features, caching=True, hash_limit=None):
        super(FirstOrderModel, self).__init__(
            output_encoder=dep.FirstOrderCoder(),
            structured_features=features,
            caching=caching,
            hash_limit=hash_limit)
        self._decoder = dep.FirstOrderDecoder(use_cache=True)

    def dynamic_program(self, x):
        problem = dep.DependencyProblem(len(x.words)-1)
        return self._decoder.dynamic_program(problem)

class ParsingPreprocessor(pydecode.preprocess.LabelPreprocessor):
    WORD = 0
    TAG = 1
    def input_labels(self, x):
        return np.array(zip(x.words, x.tags) + [("*END*", "*END*")]).T

class DepFeatures(object):
    ENC = ParsingPreprocessor
    def __init__(self):
        self._prep = self.ENC()
        self.components = [self._prep]
        self.bins = [1, 2, 3, 5, 8, 20, 40]

    def feature_templates(self, _):
        def size(t):
            return self._prep.size(t) + 1
        base = [(size(self.ENC.WORD), size(self.ENC.WORD)),
                (size(self.ENC.WORD), size(self.ENC.TAG)),
                (size(self.ENC.TAG), size(self.ENC.WORD)),
                (size(self.ENC.TAG), size(self.ENC.TAG))]

        t = base
        t += [(2, len(self.bins) + 1) + b for b in base]
        t += [(size(self.ENC.TAG), size(self.ENC.TAG),
                size(self.ENC.TAG), size(self.ENC.TAG))] * 4
        return t

    def output_features(self, x, outputs):
        p = self._prep.preprocess(x)
        o = outputs
        c = dep.FirstOrderCoder
        dist = o[:, c.HEAD] - o[:, c.MOD]
        bdist = np.digitize(dist, self.bins)
        direction = dist >= 0
        base = [(p[self.ENC.WORD][o[:, c.HEAD]], p[self.ENC.WORD][o[:, c.MOD]]),
                (p[self.ENC.WORD][o[:, c.HEAD]], p[self.ENC.TAG][o[:, c.MOD]]),
                (p[self.ENC.TAG][o[:, c.HEAD]],  p[self.ENC.WORD][o[:, c.MOD]]),
                (p[self.ENC.TAG][o[:, c.HEAD]],  p[self.ENC.TAG][o[:, c.MOD]])]
        t = base
        t += [(direction, bdist) + b  for b in base]

        for d in [(-1, -1), (-1, 1), (1,-1), (1, 1)]:
            t += [(p[self.ENC.TAG][o[:, c.HEAD] + d[0]],
                   p[self.ENC.TAG][o[:, c.HEAD]],
                   p[self.ENC.TAG][o[:, c.MOD] + d[1]],
                   p[self.ENC.TAG][o[:, c.MOD]])]
        return t, np.arange(len(t))

def train_and_save(model, X, Y, f="/tmp/weights.npy"):
    sp = StructuredPerceptron(model, verbose=1, max_iter=10, average=False)
    sp.fit(X, Y)
    np.save(f, sp.w)

DepX = namedtuple("DepX", ("words", "tags"))

def score(model, w, X, Y):
    right = []
    wrong = []
    for x, y in zip(X, Y):
        yhat = model.inference(x, w)
        right += y
        wrong += yhat
    print model.loss(right, wrong)

def read_parse(f, limit=None, length=None):
    records = nlpformat.read_csv_records(f,
                                         limit=limit,
                                         length=length)
    f = nlpformat.CONLL

    X, Y = [], []
    for record in records:
        X.append(DepX(("*ROOT*",) + tuple(record[:, f["WORD"]]),
                      ("*ROOT*",) + tuple(record[:, f["TAG"]])))
        parse = dep.DependencyParse((-1,) + tuple(map(int, record[:, f["HEAD"]])))
        Y.append(parse)
    return X, Y

def main():
    # import dowser
    # import cherrypy
    # cherrypy.engine.autoreload.unsubscribe()
    # cherrypy.config.update({'environment': 'embedded',
    #                         'server.socket_port': 8088})
    # cherrypy.tree.mount(dowser.Root())
    # cherrypy.engine.start()

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = "notebooks/data/wsj_sec_2_21_gold_dependencies"

        X, Y = read_parse(data, limit=30000, length=40)

        features = DepFeatures()
        model = FirstOrderModel(features, hash_limit=1000000)
        train_and_save(model, X, Y)
        # load_and_test(X, Y, X[5000:5600], Y[5000:5600])

if __name__ == "__main__":
    main()
