"""
Classes for sequence tagging/labeling problem.
"""
import pydecode

import itertools
import numpy as np

def all_taggings(tag_sizes):
    """
    Generate all valid tag sequences for a tagging problem.
    """
    for seq in itertools.product(*map(range, tag_sizes)):
        yield np.array(seq)


def tagger_first_order(sentence_length, tag_sizes):
    n = sentence_length
    K = tag_sizes
    t = np.max(tag_sizes)

    coder = np.arange(n * t, dtype=np.int64)\
        .reshape([n, t])
    out = np.arange(n * t * t, dtype=np.int64)\
        .reshape([n, t, t])

    c = pydecode.ChartBuilder(coder, out,
                              unstrict=True,
                              lattice=True)

    c.init(coder[0, :K[0]])
    for i in range(1, sentence_length):
        for t in range(K[i]):
            c.set_t(coder[i, t],
                    coder[i-1, :K[i-1]],
                    labels=out[i, :K[i-1], t])

    return c.finish(False)

class TaggingEncoder:
    def __init__(self, tag_sizes, order=1):
        self.tag_sizes = tag_sizes
        self.size = len(self.tag_sizes)
        self.order = order
        n = len(tag_sizes)
        t = np.max(tag_sizes)
        self.shape = (n, t, t)

    def transform(self, labels):
        return np.array(np.unravel_index(labels, self.shape)).T

    def from_path(self, path):
        parse = self.transform(path.labeling[path.labeling!=-1])
        return self.from_labels(parse)

    def to_labels(self, tagging):
        if self.order == 1:
            return np.array([[i] + tagging[i-self.order:i+1]
                             for i in range(self.order, len(tagging))])

    def from_labels(self, labels):
        sequence = np.zeros(self.size)
        for (i, pt, t) in labels:
            sequence[i] = t
            sequence[i-1] = pt
        return sequence
