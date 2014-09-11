import numpy as np
import pydecode.model
import pydecode.nlp
from collections import namedtuple

class FirstOrderModel(pydecode.model.HammingLossModel,
                      pydecode.model.DynamicProgrammingModel):

    def initialize(self, X, Y):
        self.bins = [1, 2, 3, 5, 8, 20, 40]

        # Preprocessor
        self.preprocessor = pydecode.model.Preprocessor()
        for x in X:
            for word, tag in zip(x.words, x.tags):
                self.preprocessor.add(self._preprocess_word(word, tag))
            self.preprocessor.add(self._preprocess_word("*END*", "*END*"))
        self.dp_cache_limit = 20
        super(FirstOrderModel, self).initialize(X, Y)

    def dynamic_program(self, x):
        n = len(x.words)
        if ("DP", n) not in self.cache:
            graph, encoder = pydecode.nlp.eisner(len(x.words)-1)
            self.cache["DP", n] = (graph, encoder)
            return graph, encoder
        return self.cache["DP", n]

    def part_cache(self, x):
        return len(x.words) < self.dp_cache_limit

    def _preprocess_word(self, word, tag):
        return {"WORD": word, "TAG": tag}

    def _preprocess(self, x):
        if ("PP", x) in self.cache:
            return self.cache["PP", x]
        else:
            p = self.preprocessor.transform(
                [self._preprocess_word(word, tag)
                 for word, tag in zip(x.words, x.tags)] \
                    + [self._preprocess_word("*END*", "*END*")])
            self.cache["PP", x] = p
        return p

    def parts_features(self, x, parts):
        p = self._preprocess(x)
        o = parts
        HEAD = 0
        MOD = 1
        dist = o[:, HEAD] - o[:, MOD]
        direction = dist >= 0

        bdist = np.digitize(np.abs(dist), self.bins)

        # t = [(p["TAG"][o[:, HEAD]],  p["TAG"][ o[:, MOD]])]

        # Unigram features.
        t = [(p["WORD"][o[:, HEAD]],),
             (p["WORD"][o[:, MOD]],),
             (p["POS"] [o[:, HEAD]],),
             (p["POS"] [o[:, MOD]],),
             (p["WORD"][o[:, HEAD]], p["POS"][o[:, HEAD]]),
             (p["WORD"][o[:, MOD]], p["POS"][o[:, MOD]])]
        # Bigram features.
        t += [(p["WORD"][o[:, HEAD]], p["WORD"][o[:, MOD]]),
              (p["WORD"][o[:, HEAD]], p["TAG"][ o[:, MOD]]),
              (p["TAG"][o[:, HEAD]],  p["WORD"][o[:, MOD]]),
              (p["TAG"][o[:, HEAD]],  p["TAG"][ o[:, MOD]]),
              (p["WORD"][o[:, HEAD]], p["TAG"][o[:, HEAD]],  p["TAG"][ o[:, MOD]]),
              (p["WORD"][o[:, MOD]], p["TAG"][o[:, HEAD]],  p["TAG"][ o[:, MOD]])]

        for d in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            t += [(p["TAG"][o[:, HEAD] + d[0]],
                   p["TAG"][o[:, HEAD]],
                   p["TAG"][o[:, MOD] + d[1]],
                   p["TAG"][o[:, MOD]])]

        # t = base
        t += [(direction, bdist) + b for b in t]

        return t

    def templates(self):
        def s(t):
            return self.preprocessor.size(t)
        # t = [(s("TAG"),  s("TAG"))]
        t = [(s("WORD"),),
             (s("WORD"),),
             (s("POS"),),
             (s("POS"),),
             (s("WORD"), s("POS")),
             (s("WORD"), s("POS"))]

        t += [(s("WORD"), s("WORD")),
              (s("WORD"), s("TAG")),
              (s("TAG"),  s("WORD")),
              (s("TAG"),  s("TAG")),
              (s("WORD"), s("TAG"), s("TAG")),
              (s("WORD"), s("TAG"), s("TAG"))]
        # t = base

        t += [(s("TAG"), s("TAG"),
               s("TAG"), s("TAG"))] * 4

        t += [(2, len(self.bins) + 1) + b for b in t]
        return t

DepX = namedtuple("DepX", ("words", "tags"))

def read_parse(f, limit=None, length=None):
    records = pydecode.nlp.read_csv_records(f,
                               limit=limit,
                               length=length)
    f = pydecode.nlp.CONLL
    X, Y = [], []
    for record in records:
        X.append(DepX(("*ROOT*",) + tuple(record[:, f["WORD"]]),
                      ("*ROOT*",) + tuple(record[:, f["TAG"]])))
        Y.append(np.array([-1] + [int(i) for i in record[:, f["HEAD"]]]))
    return X, Y


def main():
    from pystruct.learners import StructuredPerceptron
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = "notebooks/data/wsj_sec_2_21_gold_dependencies"
        X, Y = read_parse(data, limit=10000, length=20)

        model = FirstOrderModel(feature_hash=1000000,
                                joint_feature_format="fast")
        sp = StructuredPerceptron(model, verbose=1, max_iter=10, average=False)
        sp.fit(X, Y)
        np.save("/tmp/w", sp.w)

if __name__ == "__main__":
    main()
