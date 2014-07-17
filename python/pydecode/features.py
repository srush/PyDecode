"""
Classes for generating and working with sparse features.
"""

import numpy as np
import itertools

class FeatureGenerator(object):
    """
    Interface for generating features for a structured
    dynamic programming problem.

    Subclasses should implement `make_templates` and `generate_features`.

    """
    def make_templates(self, output_encoder):
        """
        Returns the feature template sizes.
        """
        raise NotImplementedError()

    def generate_features(self, x, outputs, output_encoder):
        """
        Generates the features for the specified outputs.

        Parameters
        ----------
        x : input

        outputs : array of output (N x I)
            An array of output structures.

        output_encoder : structured encoder

        Returns
        -------
        features : list of tuples
        """
        raise NotImplementedError()

    @property
    def components(self):
        return []

class SparseFeatureManager(object):
    """
    Helper class for generating stuctured features.
    """
    def __init__(self, feature_generator):
        self._feature_generator = feature_generator
        self._feature_templates = None
        self._feature_size = None
        self._starts = None
        self.size_features = -1

    def initialize(self, X, Y, output_coder):
        """
        Initialize the feature manager.
        """
        self._feature_templates = \
            self._feature_generator.feature_templates(output_coder)
        self._feature_size = \
            [np.product(shape)
             for shape in self._feature_templates]

        sizes = np.cumsum(self._feature_size)
        self._starts = np.insert(sizes[:-1], 0, 0)
        self.size_features = sizes[-1]
        print "FEATURE SIZE", self.size_features

    def generate_output_features(self, x, outputs):
        """
        Returns a matrix where rows correspond to an outputs
        and columns correspond to feature templates.

        Parameters
        ----------
        x : input

        outputs : array of outputs (in I)
            The output elements used to generate features.

        Returns
        ---------
        output_features : {1..d}^|outputs|x|templates|
            Value is the feature index for the corresponding
            output and template.
        """
        # Generate the feature tuples for the matrix of outputs.
        features = self._feature_generator.output_features(x, outputs)
        assert len(features) == len(self._feature_templates)

        # For each template, convert its feature tuples to
        # indices in the corresponding template space.
        rows = []
        for shape, features in itertools.izip(self._feature_templates,
                                              features):
            rows.append(np.ravel_multi_index(features, shape))

        ret = np.vstack(rows).T + self._starts

        # The final matrix is |outputs| x |templates|.
        assert ret.shape == (len(outputs),
                             len(self._feature_templates)),\
                             "%s %s" %(ret.shape, len(outputs))
        return ret

    def sparse_feature_vector(self, feature_indices):
        ind = feature_indices.ravel()
        return _SparseVec(ind)

        # self._vec = np.zeros(self.size_features)
        # vec[ind] = 1
        # return vec
        # return scipy.sparse.csc_matrix((np.repeat(1, len(ind)), ind, [0, len(ind)]),
        #                                shape=(self.size_features, 1),
        #                                dtype=np.double)

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
