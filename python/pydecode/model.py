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

class Preprocessor(object):
    def initialize(self, X):
        pass

    def preprocess(self, X):
        return X

class Pruner(object):
    def initialize(self, X, Y, out):
        pass

    def preprocess(self, x):
        return X


class StructuredCoder(object):
    def inverse_transform(self, outputs):
        pass

    def transform(self, y):
        pass

class SequencePreprocessor(object):

    def process_item(self, item):
        pass

    def size(self, typ):
        return len(self._preprocess_encoders[typ].classes_)

    def initialize(self, X):
        parts = [self.preprocess_item(item)
                 for x in X
                 for item in x]

        part_lists = zip(*parts)
        self._preprocess_encoders = []
        for part_list in part_lists:
            encoder = sklearn.preprocessing.LabelEncoder()
            encoder.fit(part_list)
            self._preprocess_encoders.append(encoder)

    def preprocess(self, x):
        part_lists = zip(*[self.preprocess_item(item) for item in x])
        return [encoder.transform(part_list)
                for part_list, encoder in
                itertools.izip(part_lists, self._preprocess_encoders)]


class DynamicProgrammingModel(StructuredModel):
    def __init__(self,
                 preprocessor=None,
                 output_coder=None,
                 pruner=None):
        self.inference_calls = 0

        self._preprocessor = preprocessor
        if preprocessor == None:
            self._preprocessor = Preprocessor()
        self._output_coder = output_coder

        self._pruner = pruner
        if pruner == None:
            self._pruner = Pruner()

    def feature_templates(self):
        raise NotImplementedError()

    def generate_features(self, element, preprocessed_x):
        raise NotImplementedError()

    def chart(self, x):
        raise NotImplementedError()

    def loss(self, yhat, y):
        raise NotImplementedError()

    def max_loss(self, y):
        raise NotImplementedError()

    def _output_features(self, output, x):
        """
        Returns
        ---------
        list of Dx1 vectors
        """
        return [array[feature]
                for array, feature in
                zip(self._feature_arrays,
                    self.generate_features(output, x))]

    def initialize(self, X, Y):
        self._preprocessor.initialize(X)
        self._pruner.initialize(X, Y, self._output_coder)

        self._feature_arrays = [np.arange(np.product(shape)).reshape(shape)
                                for shape in self.feature_templates()]

        templates = self._feature_arrays


        n_values = []
        for template in templates:
            n_values.append(template.size)

        self.encoder = sklearn.preprocessing.OneHotEncoder(
            n_values=n_values)

        features = []

        for x, y in itertools.izip(X, Y):
            preprocessed_x = self._preprocessor.preprocess(x)
            features += [self._output_features(output,
                                               preprocessed_x)
                         for output in self._output_coder.transform(y)]


        self.encoder.fit(features)
        self.size_joint_feature = self.encoder.feature_indices_[-1]

    def _feature_matrix(self, x, active_outputs, outputs):
        """
        Parameters
        ----------
        x : input structure
           Structure in X.

        active_outputs : list
           Possible outputs as indices in {1..O}

        Returns
        --------
        Dx

        mat : I x D
        """
        D = self.size_joint_feature
        A = len(active_outputs)
        I = outputs.size

        feature_arrays = self._feature_arrays

        preprocessed_x = self._preprocessor.preprocess(x)
        features = self.encoder.transform(
            [self._output_features(output, preprocessed_x)
             for output in zip(*np.unravel_index(active_outputs, outputs.shape))])
        assert features.shape == (A, D)

        # Matrix
        trans = scipy.sparse.csr_matrix(([1] * A, active_outputs,
                                         np.arange(A+1)),
                                        shape=(A, I),
                                        dtype=np.uint8)
        # print  "SPARSE time ", time.time()-a
        #D x A  A x I
        return (features.T * trans).T

    def inference(self, x, w, relaxed=False):
        """
        Parameters
        -----------
        x :

        w : D x 1 matrix

        """
        dp = self.chart(x)

        I = dp.outputs.size
        N = len(dp.hypergraph.edges)
        D = self.size_joint_feature

        assert w.shape == (D,)

        active_outputs = (dp.output_matrix * np.ones(N)).nonzero()[0]
        feature_matrix = self._feature_matrix(x, active_outputs, dp.outputs)
        assert feature_matrix.shape == (I, D)

        output_scores = w.T * feature_matrix.T
        assert output_scores.shape == (I,)

        best_outputs = pydecode.argmax(dp, output_scores)

        return self._output_coder.inverse_transform(best_outputs)

    def joint_feature(self, x, y):
        """
        Returns
        --------
        Features : D x 1 matrix
        """
        D = self.size_joint_feature

        outputs = self._output_coder.transform(y)

        #output_set = self.output_set(x)
        preprocessed_x = self._preprocessor.preprocess(x)

        cat_feats = [self._output_features(output, preprocessed_x)
                     for output in outputs]
        features = self.encoder.transform(cat_feats)

        # Sum up the features.
        final_features = features.T * np.ones(features.shape[0]).T
        assert final_features.shape == (D, )
        return final_features
