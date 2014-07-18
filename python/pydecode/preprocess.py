"""
Classes for preprocessing input data.
"""

import numpy as np
from collections import defaultdict
from itertools import izip

class LabelPreprocessor(object):
    """
    Preprocess an input structure into a matrix of enumerated labels.

    For instance in POS tagging, labels for each word might be
    prefixes, suffixes, capitalization, etc. These can then be easily
    combined into features.
    """
    def input_labels(self, x):
        """
        Return the labels for the given input as a matrix.

        Returns
        -------
        matrix : m x |labels|-matrix (of any type)
            Size is the number of labels. m is specific to x.
            (e.g number of words).
        """
        raise NotImplementedError()

    def __init__(self):
        self._encoders = []
        self._cache = {}

    def size(self, label_num):
        """
        The number of elements in label_num.

        Parameters
        ----------
        label_num : int
           The label number.
        """
        return len(self._encoders[label_num])

    def inverse_transform_label(self, label_num, label_ind):
        return self._rev_encoders[label_num][label_ind]

    def initialize(self, X, _, __):
        labels = np.hstack([self.input_labels(x) for x in X])
        self._encoders = [defaultdict(lambda: 0)
                          for _ in range(labels.shape[0])]

        self._rev_encoders = [defaultdict(lambda: "UNK")
                              for _ in range(labels.shape[0])]

        for label, enc, rev in izip(labels, self._encoders, self._rev_encoders):
            ls = list(set(label))
            enc.update(zip(ls, range(1, len(ls)+1)))
            rev.update(zip(range(1, len(ls)+1), ls))

        for x in X:
            self._cache[tuple(x)] = self.preprocess(x)

    def preprocess(self, x):
        """
        Return the processed labels for the given input as a matrix.

        Returns
        -------
        matrix : m x |labels| matrix of ints
            Size is the number of labels. m is specific to x.
            (e.g number of words).

        """
        v = self._cache.get(tuple(x), None)
        if v is not None:
            return v
        labels = self.input_labels(x)
        return np.array([[enc[l] for l in ls]
                         for enc, ls in izip(self._encoders, labels)])
