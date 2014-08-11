import numpy as np
from pystruct.models import StructuredModel
import pydecode
from collections import defaultdict
import scipy
from copy import copy

class _FastVec(object):
    __array_priority__ = 2

    def __init__(self, values):
        self.d = {}
        for v in values:
            self.d.setdefault(v, 0)
            self.d[v] += 1

    def __repr__(self):
        return repr(self.d)

    def __setitem__(self, key, val):
        self.d[key] = val

    def __getitem__(self, key):
        return self.d[key]

    def __sub__(self, other):
        new = _FastVec([])
        new.d = copy(self.d)
        for k, val in other.d.iteritems():
            if k in new.d:
                if new.d[k] == val: del new.d[k]
                else: new.d[k] -= val
            else: new.d[k] = -val
        return new

    def __rmul__(self, other):
        new = _FastVec([])
        new.d = copy(self.d)
        for k, val in self.d.iteritems():
            new.d[k] *= other
        return new

    def __radd__(self, other):
        other.T[:, self.d.keys()] += self.d.values()
        return other


def sparse_feature_indices(label_features, templates, feature_hash=None):
    """
    feature_matrix : int ndarray

    templates : list of tuples
    """
    rows = []
    offsets = [0]
    for shape in templates[:-1]:
        offsets.append(offsets[-1] + np.product(shape))

    for features, shape in zip(label_features, templates):
        out = np.ravel_multi_index(features, shape)
        rows.append(out)

    feature_matrix = np.vstack(rows).T + offsets
    if feature_hash is not None:
        feature_matrix = feature_matrix % feature_hash
    return feature_matrix

def sparse_vector(indices, shape):
    ind = indices.ravel()
    return scipy.sparse.csc_matrix(
        (np.repeat(1, len(ind)), ind, [0, len(ind)]),
        shape=shape,
        dtype=np.double)


def dense_vector(indices, shape):
    arr = np.zeros(shape)
    arr[indices] += 1
    return arr

def fast_vector(indices, shape):
    return _FastVec(indices.ravel())


class Preprocessor:
    def __init__(self):
        self.dict = defaultdict(lambda: defaultdict(lambda: 0))

    def size(self, key):
        return len(self.dict[key]) + 1

    def add(self, d):
        for key in d:
            if d[key] not in self.dict[key]:
                self.dict[key][d[key]] = \
                    len(self.dict[key]) + 1

    def transform(self, d_list):
        preprocessed = {}
        for key in self.dict:
            preprocessed[key] = np.zeros(len(d_list), dtype=np.int64)

        for i, d in enumerate(d_list):
            for key in d:
                preprocessed[key][i] = \
                    self.dict[key][d[key]]
        return preprocessed

class DynamicProgrammingModel(StructuredModel):
    def __init__(self, feature_hash=None, use_cache=True,
                 joint_feature_format=None):
        # Necessary for pystruct.
        self.inference_calls = 0
        self.feature_hash = feature_hash
        self.cache = {}
        self._use_cache = use_cache
        self.joint_feature_format = joint_feature_format

    def templates(self):
        raise NotImplementedError()

    def parts_features(self, x, labels):
        raise NotImplementedError()

    def dynamic_program(self, x):
        raise NotImplementedError()

    def loss(self, yhat, y):
        raise NotImplementedError()

    def max_loss(self, y):
        raise NotImplementedError()

    def initialize(self, X, Y):
        """
        Initialize the model and components based on the training data.
        """
        feature_size = \
            np.sum([np.product(template)
                    for template in self.templates()])
        if self.feature_hash is not None:
            feature_size = min(self.feature_hash, feature_size)
        self.size_joint_feature = (feature_size, 1)

    def joint_feature(self, x, y):
        if ("FEAT", x, tuple(y)) in self.cache:
            return self.cache["FEAT", x, tuple(y)]

        _, encoder = self.dynamic_program(x)
        labels = encoder.transform_structure(y)
        parts_features = self.parts_features(x, labels)
        feature_matrix = sparse_feature_indices(parts_features,
                                                self.templates(),
                                                self.feature_hash)

        if self.joint_feature_format is None:
            feature_vector = dense_vector(feature_matrix,
                                          self.size_joint_feature)
        elif self.joint_feature_format == "sparse":
            feature_vector = sparse_vector(feature_matrix,
                                           self.size_joint_feature)
        elif self.joint_feature_format == "fast":
            feature_vector = fast_vector(feature_matrix,
                                         self.size_joint_feature)

        if self._use_cache:
            self.cache["FEAT", x, tuple(y)] = feature_vector
        return feature_vector

    def inference(self, x, w, relaxed=False):
        graph, encoder = self.dynamic_program(x)
        if ("IND", x) in self.cache:
            feature_indices = self.cache["IND", x]
        else:
            graph_labels = graph.labeling[graph.labeling != -1]
            parts = encoder.transform_labels(graph_labels)
            parts_features = self.parts_features(x, parts)
            feature_indices = sparse_feature_indices(parts_features,
                                                     self.templates(),
                                                     self.feature_hash)
            if self._use_cache:
                self.cache["IND", x] = feature_indices

        # Sum the feature weights for the features in each label row.
        label_weights = np.zeros(len(graph.labeling))
        label_weights[graph.labeling != -1] = \
            np.sum(np.take(w, feature_indices, mode="clip"), axis=1)

        # weights = pydecode.transform(graph, label_weights)
        path = pydecode.best_path(graph, label_weights)
        y = encoder.transform_path(path)
        return y

class HammingLossModel(object):
    def loss(self, y, yhat):
        assert(len(y) == len(yhat))
        match = 0
        for y1, y2 in zip(yhat, y):
            if y1 == y2:
                match += 1
        return 1.0 - (match / float(len(y)))

    def max_loss(self, y):
        return 1.0

class ZeroOneLossModel(object):
    def loss(self, y, yhat):
        return 1.0 if y != yhat else 0.0

    def max_loss(self, y):
        return 1.0
