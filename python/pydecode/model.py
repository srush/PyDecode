"""
A structured prediction and training library.
Requires pystruct.
"""

from pystruct.models import StructuredModel
import sklearn.preprocessing
import itertools
import scipy.sparse
import pydecode
import numpy as np
import pydecode.lp as lp
from collections import defaultdict
from copy import copy

class StructuredCoder(object):
    def inverse_transform(self, outputs):
        raise NotImplementedError()

    def transform(self, y):
        raise NotImplementedError()

class LabelPreprocessor(object):
    def input_labels(self, x):
        raise NotImplementedError()

    def size(self, typ):
        return len(self._encoders[typ])

    def initialize(self, X):
        labels = [self.input_labels(x) for x in X]
        labels = np.hstack(labels)
        self._encoders = []
        for i in range(labels.shape[0]):
            encoder = defaultdict(lambda: 0)
            encoder.update([(j, i) for i, j in enumerate(set(labels[i]), 1)])
            self._encoders.append(encoder)

            #     #sklearn.preprocessing.LabelEncoder().fit(labels[i]))
            # # Grr.. Label encoder doesn't handle oov.
            # self._encoders[-1].classes_ = \
            #     np.append(self._encoders[-1].classes_, '<unknown>')

        self._cache = {}
        for x in X:
            self._cache[tuple(x)] = self.preprocess(x)

    def preprocess(self, x):
        v = self._cache.get(tuple(x), None)
        if v is not None:
            return v
        labels = self.input_labels(x)
        return np.array([[self._encoders[i][l] for l in labels[i]]
                         for i in range(labels.shape[0])])


class _SparseVec(object):
    __array_priority__ = 2

    def __init__(self, values):
        self.d = {}
        for v in values:
            self.d[v] = 1

    def __repr__(self):
        return repr(self.d)

    def __setitem__(self, key, val):
        self.d[key] = val

    def __getitem__(self, key):
        return self.d[key]

    def __sub__(self, other):
        new = _SparseVec([])
        new.d = copy(self.d)
        for k, val in other.d.iteritems():
            if k in new.d:
                if new.d[k] == val: del new.d[k]
                else: new.d[k] -= val
            else: new.d[k] = -val
        return new

    def __rmul__(self, other):
        new = _SparseVec([])
        new.d = copy(self.d)
        for k, val in self.d.iteritems():
            new.d[k] *= other
        return new

    def __radd__(self, other):
        other.T[:, self.d.keys()] += self.d.values()
        return other

class StructuredFeatures(object):
    def output_features(self, x, outputs):
        raise NotImplementedError()

    @property
    def feature_templates(self):
        raise NotImplementedError()

    @property
    def size_joint_feature(self):
        return (self.encoder.feature_indices_[-1], 1)

    @property
    def size_features(self):
        return self.encoder.feature_indices_[-1]

    def __init__(self, output_coder):
        self.output_coder = output_coder
        self._feature_templates = self.feature_templates
        self._feature_size = [np.product(shape)
                              for shape in self._feature_templates]
        self._starts = np.cumsum(self._feature_size)[:-1]
        self._starts = np.insert(self._starts, 0, 0)
        self.encoder = None
        self._joint_feature_cache = {}

    def initialize(self, X, Y):
        # Collect all the active features.
        features = []
        # Make a one-hot encoder to map features to indices.
        self.encoder = sklearn.preprocessing.OneHotEncoder(
            n_values=[size
                      for size in self._feature_size])

        for x, y in itertools.izip(X, Y):
            outputs = self.output_coder.transform(y)
            features.append(self._stack_output_features(x, outputs))

        self.encoder.fit(np.vstack(features))

        for x, y in itertools.izip(X, Y):
            outputs = self.output_coder.transform(y)
            feature_indices = \
                self._stack_output_features(x, outputs, shift=True)

            self._joint_feature_cache[x, y] = \
                self._sparse_feature_vector(feature_indices)
        print "FEATURE SIZE", self.encoder.feature_indices_[-1]

    def _stack_output_features(self, x, outputs, shift=False):
        """
        Returns a matrix where rows correspond to an outputs
        and columns correspond to feature templates.

        Returns
        ---------
        |outputs| x |templates| matrix, value is the index in the template.
        """
        # Generate the feature tuples for the matrix of outputs.
        features = self.output_features(x, outputs)
        assert len(features) == len(self._feature_templates)

        # For each template, convert its feature tuples to
        # indices in the corresponding template space.
        rows = []
        for shape, features in itertools.izip(self._feature_templates,
                                              features):
            rows.append(np.ravel_multi_index(features, shape))

        if shift:
            ret = np.vstack(rows).T + self._starts
        else:
            ret = np.vstack(rows).T

        # The final matrix is |outputs| x |templates|.
        assert ret.shape == (len(outputs),
                             len(self._feature_templates)),\
                             "%s %s" %(ret.shape, len(outputs))
        return ret

    def _sparse_feature_vector(self, feature_indices):
        ind = feature_indices.ravel()
        # self._vec = np.zeros(self.size_features)
        # vec[ind] = 1
        # return vec
        # return scipy.sparse.csc_matrix((np.repeat(1, len(ind)), ind, [0, len(ind)]),
        #                                shape=(self.size_features, 1),
        #                                dtype=np.double)

        return _SparseVec(ind)

    def joint_features(self, x, y):
        """
        Parameters
        ----------
        x : input structure
           Structure in X.

        y : output structure

        Returns
        --------
        mat : D- sparse matrix
        """
        cached_features = self._joint_feature_cache.get((x, y), None)
        if cached_features is not None:
            return cached_features
        outputs = self.output_coder.transform(y)
        feature_indices = \
            self._stack_output_features(x, outputs, shift=True)
        return self._sparse_feature_vector(feature_indices)



class DynamicProgrammingModel(StructuredModel):
    def __init__(self,
                 output_coder,
                 structured_features):
        self.inference_calls = 0
        self._output_coder = output_coder
        self._structured_features = structured_features
        self._dp_cache = {}
        self._feature_cache = {}

    def dynamic_program(self, x):
        raise NotImplementedError()

    def loss(self, yhat, y):
        raise NotImplementedError()

    def max_loss(self, y):
        raise NotImplementedError()

    def initialize(self, X, Y):
        self._structured_features.initialize(X, Y)
        self.size_joint_feature = \
            self._structured_features.size_joint_feature

        for x in X:
            dp = self.dynamic_program(x)
            import sys

            # print len(dp.hypergraph.edges), dp.outputs.nbytes, dp.output_indices.nbytes, dp.items.nbytes, dp.item_indices.nbytes

            self._dp_cache[x] = dp

            feature_indices = \
                self.feature_indices(x, dp, dp.active_outputs)
            self._feature_cache[x] = feature_indices

    def joint_feature(self, x, y):
        return self._structured_features.joint_features(x, y)

    def feature_indices(self, x, dp, output_indices):
        outputs = np.array(np.unravel_index(output_indices,
                                            dp.outputs.shape)).T
        return self._structured_features._stack_output_features(x, outputs, shift=True)

    def argmax(self, x, w):
        dp = self._dp_cache.get(x, None)
        if dp is None:
            dp = self.dynamic_program(x)
            # self._dp_cache[x] = dp

        indices = self._feature_cache.get(x, None)
        if indices is None:
            feature_indices = \
                self.feature_indices(x, dp, active_indices)
        else:
            feature_indices = indices
        active_scores = \
            np.sum(np.take(np.asarray(w), feature_indices, mode="clip"), axis=1)

        potentials = pydecode.map_active_potentials(dp, active_scores)
        path = pydecode.best_path(dp.hypergraph, potentials)

        # Compute features.
        indices = dp.active_output_indices.take(path.edge_indices)
        features = feature_indices[indices[indices != -1]]

        # For testing
        # assert(potentials * path.v == np.sum(w.take(features.ravel())))

        return pydecode.path_output(dp, path), features

    def inference(self, x, w, relaxed=False):
        """
        Parameters
        -----------
        x :

        w : D x 1 matrix

        """
        best_outputs, features = self.argmax(x, w)
        y = self._output_coder.inverse_transform(best_outputs)
        self._structured_features._joint_feature_cache[x, y] = \
            self._structured_features._sparse_feature_vector(features)
        return y
