"""
Classes for sequence tagging/labeling problem.
"""
import pydecode
import itertools
import numpy as np
from pydecode.encoder import StructuredEncoder


def tagger_first_order(sentence_length, tag_sizes):
    n = sentence_length
    K = tag_sizes
    t = np.max(tag_sizes)

    coder = np.arange(n * t, dtype=np.int64)\
        .reshape([n, t])
    part_encoder = TaggingEncoder(tag_sizes, 1)
    out = part_encoder.encoder

    c = pydecode.ChartBuilder(coder, out,
                              unstrict=True,
                              lattice=True)

    c.init(coder[0, :K[0]])
    for i in range(1, sentence_length):
        for t in range(K[i]):
            c.set_t(coder[i, t],
                    coder[i-1, :K[i-1]],
                    labels=out[i, :K[i-1], t])

    return c.finish(False), part_encoder

class TaggingEncoder(StructuredEncoder):
    def __init__(self, tag_sizes, order=1):
        self.tag_sizes = tag_sizes
        self.size = len(self.tag_sizes)
        self.order = order
        n = len(tag_sizes)
        t = np.max(tag_sizes)
        shape = (n, t, t)
        super(TaggingEncoder, self).__init__(shape)

    def transform_structure(self, tagging):
        if self.order == 1:
            return np.array([np.append([i], tagging[i-self.order:i+1])
                             for i in range(self.order, len(tagging))])

    def from_parts(self, parts):
        sequence = np.zeros(self.size, dtype=np.int32)
        for (i, pt, t) in parts:
            sequence[i] = t
            sequence[i-1] = pt
        return sequence

    def all_structures(self):
        """
        Generate all valid tag sequences for a tagging problem.
        """
        for seq in itertools.product(*map(range, self.tag_sizes)):
            yield np.array(seq)

    def random_structure(self):
        sequence = np.zeros(len(self.tag_sizes))
        for i, size in enumerate(self.tag_sizes):
            sequence[i] = np.random.randint(size)
        return sequence
