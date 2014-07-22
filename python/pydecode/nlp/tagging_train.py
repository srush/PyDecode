import pydecode
import pydecode.model
from pydecode.preprocess import LabelPreprocessor
import pydecode.nlp.tagging
import pydecode.nlp.format as nlpformat
import sklearn
import sklearn.metrics
import numpy as np
from pydecode.nlp.tagging \
    import PrunedBigramTagger, BiTagEncoder, DictionaryPruner, BigramTagger
from pystruct.learners import StructuredPerceptron

class BigramModel(pydecode.model.HammingLossModel,
                  pydecode.model.DynamicProgrammingModel):
    def __init__(self, features, caching=True):
        super(BigramModel, self).__init__(output_encoder=BiTagEncoder(),
                                          structured_features=features,
                                          caching=caching)
        self._decoder = BigramTagger(True)

    def dynamic_program(self, x):
        problem = pydecode.nlp.tagging.TaggingProblem(
            len(x),
            tag_sizes=[1] + [self._encoder.tags] * (len(x)-2) + [1])
        return self._decoder.dynamic_program(problem)

class PrunedBigramModel(pydecode.model.HammingLossModel,
                  pydecode.model.DynamicProgrammingModel):
    def __init__(self, features, pruner=None, caching=True):
        super(BigramModel, self).__init__(
            output_encoder=BiTagEncoder(),
            structured_features=features,
            components=[pruner],
            caching=caching)
        self.pruner = pruner
        self._decoder = PrunedBigramTagger(self._encoder.tags)

    def dynamic_program(self, x):
        problem = pydecode.nlp.tagging.TaggingProblem(
            len(x), None,
            pruned_tag_sizes=self.pruner.table(x))
        return self._decoder.dynamic_program(problem)

class SimpleBigramFeatures(object):
    def feature_templates(self, output_coder):
        c = output_coder
        return [(c.tags,),
                (c.tags, c.tags)]

    def output_features(self, x, outputs, output_coder):
        c = output_coder
        return [(outputs[:, c.TAG],),
                (outputs[:, c.TAG], outputs[:, c.PREV_TAG])]

class TaggingPreprocessor(LabelPreprocessor):
    WORD = 0
    PREFIX_1 = 1
    PREFIX_2 = 2
    PREFIX_3 = 3
    SUFFIX_1 = 4
    SUFFIX_2 = 5
    SUFFIX_3 = 6

    def input_labels(self, x):
        labels = []
        for word in x:
            labels.append([word, word[:1], word[:2], word[:3],
                           word[-3:], word[-2:], word[-1:]])
        return np.array(labels).T

class BetterBigramFeatures(object):
    ENC = TaggingPreprocessor

    def __init__(self):
        self._prep = TaggingPreprocessor()
        self.components = [self._prep]

    def feature_templates(self, output_coder):
        tags = output_coder.tags
        def size(t):
            return self._prep.size(t) + 1
        return [(tags, size(self.ENC.WORD)),
                (tags, size(self.ENC.SUFFIX_1)),
                (tags, size(self.ENC.SUFFIX_2)),
                (tags, size(self.ENC.SUFFIX_3)),
                (tags, size(self.ENC.PREFIX_1)),
                (tags, size(self.ENC.PREFIX_2)),
                (tags, size(self.ENC.PREFIX_3)),
                (tags, tags)]

    def output_features(self, x, outputs):
        p = self._prep.preprocess(x)
        c = BiTagEncoder
        pos = outputs[:, c.POS]
        tag = outputs[:, c.TAG]
        t =  [(tag, p[self.ENC.WORD][pos]),
              (tag, p[self.ENC.SUFFIX_1][pos]),
              (tag, p[self.ENC.SUFFIX_2][pos]),
              (tag, p[self.ENC.SUFFIX_3][pos]),
              (tag, p[self.ENC.PREFIX_1][pos]),
              (tag, p[self.ENC.PREFIX_2][pos]),
              (tag, p[self.ENC.PREFIX_3][pos]),
              (tag, outputs[:,c.PREV_TAG])]
        return t


def train_and_save(model, X, Y, f="/tmp/weights.npy"):
    sp = StructuredPerceptron(model, verbose=1, max_iter=10, average=True)
    sp.fit(X, Y)
    np.save(f, sp.w)

def load_and_test(model, X_test, Y_test, f="/tmp/weights.npy"):
    w = np.load(f)
    total = []
    correct = []
    for x, y in zip(X_test, Y_test):
        total += list(model.inference(x, w))
        correct += list(y)
    print sklearn.metrics.hamming_loss(all, correct)

def read_tags(f, limit, length):
    data = "notebooks/data/wsj_sec_2_21_gold_dependencies"
    records = nlpformat.read_csv_records(data, limit=limit, length=length)
    f = nlpformat.CONLL
    return zip(*[(("*START*",) + tuple(record[:, f["WORD"]]) + ("*END*",),
                  ("*START*",) + tuple(record[:, f["TAG"]]) + ("*START*",))
                 for record in records])

def score(model, w, X, Y):
    right = []
    wrong = []
    for x, y in zip(X, Y):
        yhat = model.inference(x, w)
        right += y
        wrong += yhat
    print model.loss(right, wrong)

def main():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X, Y = read_tags("notebooks/data/wsj_sec_2_21_gold_dependencies")
        features = BetterBigramFeatures()
        model = BigramModel(features, DictionaryPruner())

        # X = X[:1000]
        # Y = Y[:1000]
        train_and_save(model, X[:500], Y[:500], "q_test")

if __name__ == "__main__":
    main()

        # data_X = [["*START*"] + sentence.split() + ["*END*"]  for sentence in
        #           ["the dog walked",
        #            "in the park",
        #            "in the dog"]]
        # data_Y =  [["*START*"] + tags.split() + ["*START*"]
        #            for tags in ["D N V", "I D N", "I D N"]]
        # coder = pydecode.nlp.tagging.BiTagCoder()
        # coder.fit(data_Y)
        # model = BigramModel(coder, SimpleBigramFeatures(coder))
        # sp = StructuredPerceptron(model, verbose=1, max_iter=10)
        # sp.fit(data_X, data_Y)


        # prep = TaggingPreprocessor()
        # prep.initialize(data_X)
        # model = BigramModel(coder, BetterBigramFeatures(coder, prep))
        # sp = StructuredPerceptron(model, verbose=0, max_iter=10)
        # sp.fit(data_X, data_Y)
        # data = "notebooks/data/wsj_sec_2_21_gold_dependencies"
        # records = format.read_csv_records(data, [], [])

        # # sents = [zip(*sentence) for sentence in conll_sentences(
        # # sents = [zip(*sentence) for sentence in sentences("notebooks/data/tag_train_small.dat")]

        # f = format.CONLL
        # X, Y = zip(*[(("*START*",) + tuple(record[:, f["WORD"]]) + ("*END*",),
        #               ("*START*",) + tuple(record[:, f["TAG"]]) + ("*START*",))
        #              for record in records])

        # features = BetterBigramFeatures()
        # model = BigramModel(features, DictionaryPruner())

        # # X = X[:1000]
        # # Y = Y[:1000]
        # train_and_save(model, X[:1000], Y[:1000], "q_test")
        # # train_and_save(X[:39000], Y[:39000], "q")
        # # load_and_test(X, Y, X[5000:5600], Y[5000:5600])
