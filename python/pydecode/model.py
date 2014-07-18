from __future__ import division, absolute_import

"""
Classes for training structured models based on dynamic programming.
Focuses particularly on large training sets with high-dimensional
sparse binary features which are common in natural language processing.

Michael Collins
Discriminative training methods for hidden markov models: Theory and
experiments with perceptron algorithms, 2002

Training algorithms use PyStruct.

Andreas C. Mueller, Sven Behnke
PyStruct - Structured prediction in Python, 2014
http://pystruct.github.io/
"""

from collections import defaultdict
from copy import copy
from pystruct.models import StructuredModel
import itertools
import numpy as np
import pydecode
import scipy.sparse
import pydecode.features

class _Cache:
    """
    Helper class for caching.
    """
    def __init__(self):
        self._d = {}

    def __setitem__(self, key, val):
        self._d[key] = val

    def __getitem__(self, key):
        return self._d[key]

    def get(self, key, default_fn):
        """
        Lazy version of dictionary .get(key, default).
        Evaluates default_fn only if key is not in dict.
        """
        val = self._d.get(key, None)
        if val is not None:
            return val
        else:
            return default_fn()

class DynamicProgrammingModel(StructuredModel):
    """
    Main class for training dynamic programming models with
    argmax-style training (i.e. perceptron, struct-svm algorithms).

    Subclasses must implements `loss`, `maxloss`, and
    `dynamic_program`, and provide a `FeatureGenerator` and
    `OutputEncoder`.
    """
    def __init__(self,
                 output_encoder,
                 structured_features,
                 components=None,
                 caching=True,
                 hash_limit=None):
        """
        Parameters
        ----------
        output_encoder : `OutputEncoder`
            Used to tranform results `y` to outputs.

        structured_features : `FeatureGenerator`
            Used to generate features from outputs.

        components : list of objects
            Additional components associated
            with the model, for instace pruners.  Initialized
            and stored as part of the model.

        caching : bool (defaults to true)
            Store dynamic programming object and features for training.

        hash_limit : int or none (defaults to none)
            Limit the feature size by hashing.

        """

        # Necessary for pystruct.
        self.inference_calls = 0

        self._encoder = output_encoder
        self._feature = \
            pydecode.features.SparseFeatureManager(structured_features,
                                                   hash_limit)

        self._structured_features = structured_features

        # Initialize caching.
        self._use_cache = caching

        # Stores dp for x.
        self._dp_cache = _Cache()

        # Stores active feature indices for x.
        self._feature_cache = _Cache()
        self._feature_matrix_cache = _Cache()

        # Stores joint feature vectors for x, y.
        self._joint_feature_cache = _Cache()
        self._joint_feature_inference_cache = _Cache()

        # Component storage.
        self._components = structured_features.components
        self._components += [self._feature]
        if components is not None:
            self._components += components

        # if structured_features.components is not None:
        #     self._components += structured_features.components


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
        # Initialize the encoder.
        self._encoder.fit(Y)

        # for component in structured_features.components:
        #     component.initialize(X, Y)

        # Initialize the components.
        for component in self._components:
            component.initialize(X, Y, self._encoder)

        # Set the features size (required for pystruct).
        self.size_joint_feature = \
            (self._feature.size_features, 1)

        # If using caching, prime all the caches.
        # (This takes most of the time, so add progress bar).
        if self._use_cache:
            for x, y in itertools.izip(X, Y):
                dp = self.dynamic_program(x)
                self._dp_cache[x] = dp
                # self._feature_cache[x] = self._active_feature_indices(x, dp)
                # self._feature_matrix_cache[x] = \
                #     self._feature_sparse_matrix(self._feature_cache[x])
                self._joint_feature_cache[x, y] = self._joint_feature(x, y)


    def _joint_feature(self, x, y):
        """
        Generates joint feature vectors.
        """
        outputs = self._encoder.transform(y)
        feature_indices = \
            self._feature.generate_output_features(x, outputs)
        return self._feature.sparse_feature_vector(feature_indices)

    def joint_feature(self, x, y):
        """
        Computes a joint feature vector for (x, y) pair.

        Returns
        --------
        mat : D- sparse matrix
        """
        return self._joint_feature_cache.\
            get((x, y),
                lambda: self._joint_feature_inference_cache.\
                    get((x, y),
                        self._joint_feature(x, y)))

    def _active_feature_indices(self, x, dp):
        """
        Generates the features for each output that is seen in
        the dynamic program (active_output).

        Returns
        -------
        matrix : |dp.active_indices|x|templates| - matrix
            A matrix for feature indices for each active output.
        """
        return self._feature.generate_output_features(
            x, dp.active_output_elements)

    def _feature_sparse_matrix(self, feature_indices):

        return scipy.sparse.csc_matrix(
            (np.ones_like(feature_indices.ravel()),
             feature_indices.ravel(),
             np.arange(feature_indices.shape[0] + 1) *
             feature_indices.shape[1]),
            shape=(self.size_joint_feature[0],
                   feature_indices.shape[0]),
            dtype=np.uint8)


    def _active_scores(self, w, dp, feature_indices, product=False):
        """
        This is the main feature 'dot-product'. In theory we have a
        dx1-parameter vector w and a |I|xd-feature vector F. We would
        like to compute F w to get a score for each element in
        output I.

        For efficiency we use two optimizations (1) instead of using
        the full set I, we only consider the outputs that appear in
        the DP, i.e. active outputs A \subset I. (2) to avoid
        constructing F explicitly we construct an index matrix
        {1..d}^|A|x|T| where T is the set of feature templates.
        This is feature_indices, see features.py for this code.

        After this dot-product `active_scores` is a R^|A| vector
        that has the score for each active output index.
        """
        fn = np.sum
        if product:
            fn = np.product
        return \
            fn(np.take(np.asarray(w), feature_indices,
                           mode="clip"), axis=1)


    def inference(self, x, w, relaxed=False):
        """
        Computes the best output y based on input x and
        weight parameter vector w.

        Parameters
        -----------
        x : input

        w : dx1 - parameter matrix

        Returns
        -------
        y : output
           The best output structure.

           Returns ::math::`\arg\max_y (F_x w) (E y)`
           where F_x is a feature matrix, E is the output encoder.
        """
        # First get the dp and active features from the cache.
        dp = self._dp_cache.get(x, lambda: self.dynamic_program(x))
        feature_indices = self._feature_cache\
            .get(x, lambda: self._active_feature_indices(x, dp))

        active_scores = self._active_scores(w, dp, feature_indices)

        # Use the active score vector to find the best path and
        # transform to a result y.
        scores = pydecode.map_active_potentials(dp, active_scores)
        path = pydecode.best_path(dp.hypergraph, scores)
        best_outputs = pydecode.path_output(dp, path)
        y = self._encoder.inverse_transform(best_outputs)

        if self._use_cache:
            # If we are caching, we will likely need the features of
            # this (x, y) pair. Since we already have the path, it is
            # better to cache them now.
            indices = dp.active_output_indices.take(path.edge_indices)
            features = feature_indices[indices[indices != -1]]
            self._joint_feature_inference_cache = {}
            self._joint_feature_inference_cache[x, y] = \
                self._feature.sparse_feature_vector(features)

            # For testing:
            # assert(potentials * path.v == np.sum(w.take(features.ravel())))
        return y

    def expected_counts(self, x, exp_w, m):
        dp = self._dp_cache.get(x, lambda: self.dynamic_program(x))
        feature_indices = self._feature_cache\
            .get(x, lambda: self._active_feature_indices(x, dp))

        feature_matrix = self._feature_matrix_cache\
            .get(x, lambda: None)

        active_scores = self._active_scores(exp_w, dp, feature_indices,
                                            product=True)
        scores = pydecode.map_active_potentials(dp, active_scores)


        node_marg, edge_marginals = \
            pydecode.marginals(dp.hypergraph, scores,
                               kind=pydecode.Inside)
        norm = node_marg[dp.hypergraph.root.id]

        self._set_marg(dp, norm, edge_marginals, feature_indices, feature_matrix, m)
        return np.log(norm)

    def _set_marg(self, dp, norm, edge_marginals, feature_indices, matrix, m):
        # Assume active outputs are unique. (Projection)
        active_output_marg = dp.active_output_matrix * edge_marginals
        normed_active_marg = active_output_marg / norm

        # This is a more efficienct way of writing.
        # for active, marg in enumerate(normed_active_marg):
        #     m[feature_indices[active]] += marg
        m += (matrix * normed_active_marg)[:, np.newaxis]

        # np.add.at(m, feature_indices,
        #           normed_active_marg[:, np.newaxis, np.newaxis])



        # from numpy.testing import assert_array_almost_equal
        # # assert_array_almost_equal(m2,m3)


        # for j in range(feature_indices.shape[1]):
        #     # print m2[feature_indices[:, j]].squeeze().shape
        #     # print normed_active_marg.reshape
        #     np.add.at(m3, feature_indices[:, j], normed_active_marg[:, np.newaxis])
        #     print j

        # # # print m2[feature_indices].shape
        # # # print normed_active_marg.shape

        # normed_edge = np.exp(edge_marginals - norm)
        # for edge, marg in enumerate(normed_edge):
        #     active = dp.active_output_indices[edge]
        #     if active != -1:
        #         m3[feature_indices[active]] += marg

        # assert_array_almost_equal(m2,m3)


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
