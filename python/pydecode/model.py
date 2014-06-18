"""
A structured prediction and training library.
Requires pystruct.
"""

from pystruct.models import StructuredModel
import sklearn.preprocessing
from sklearn.feature_extraction import DictVectorizer
import itertools
import scipy.sparse
import pydecode
import numpy as np
import time
import pydecode.lp as lp
import pulp
import sys

class Preprocessor(object):
    def initialize(self, X):
        pass

    def preprocess(self, X):
        return X

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

class Pruner(object):
    def initialize(self, X, Y, out):
        pass

    def preprocess(self, x):
        return X

class DynamicProgrammingModel(StructuredModel):
    def __init__(self,
                 preprocessor=None,
                 output_coder=None,
                 pruner=None):
        self._vec = DictVectorizer()
        self._debug = False
        self.inference_calls = 0

        self._preprocessor = preprocessor
        if preprocessor == None:
            self._preprocessor = Preprocessor()
        self._output_coder = output_coder


        self._pruner = pruner
        if pruner == None:
            self._pruner = Pruner()

    def feature_templates(self):
        pass

    def generate_features(self, element, preprocessed_x):
        pass

    def chart(self, x):
        pass

    def _chart(self, x):
        pass

    def element_features(self, element, templates, preprocessed_x):
        """
        Returns
        ---------
        list of Dx1 vectors
        """
        return [template.index(feature)
                for template, feature in zip(templates,
                                             self.generate_features(element, preprocessed_x))]

    def initialize(self, X, Y):
        self._preprocessor.initialize(X)
        self._pruner.initialize(X, Y, self._output_coder)

        templates = self._wrapped_feature_templates()


        n_values = []
        for template in templates:
            n_values.append(len(template))

        self.encoder = sklearn.preprocessing.OneHotEncoder(
            n_values=n_values)

        features = []

        templates = self._wrapped_feature_templates()
        for x, y in itertools.izip(X, Y):
            preprocessed_x = self._preprocessor.preprocess(x)
            features += [self.element_features(element, templates,
                                               preprocessed_x)
                         for element in self._output_coder.transform(y)]

        #mat = scipy.sparse.vstack(features)

        self.encoder.fit(features)
        self.size_joint_feature = self.encoder.feature_indices_[-1]
        # self.output_parts = sklearn.preprocessing.LabelEncoder()
        # self.output_parts.fit(outputs)

    def _wrapped_feature_templates(self):
        return map(pydecode.IndexSet, self.feature_templates())

    def _feature_matrix(self, x, mat, output_set):
        """
        Returns
        --------
        Dx

        mat : I x D
        """
        a = time.time()
        D = self.encoder.feature_indices_[-1]
        I, N = mat.shape

        # I = len(output_set)

        templates = self._wrapped_feature_templates()
        features = []
        preprocessed_x = self._preprocessor.preprocess(x)

        active_outputs = (mat * np.ones(N)).nonzero()[0]
        A = len(active_outputs)
        # print  "INIT time ", time.time()-a

        a = time.time()
        features = self.encoder.transform(
            [self.element_features(element, templates, preprocessed_x)
             for element in output_set.elements(active_outputs)])
        assert features.shape == (A, D)
        # print  "ENCODE time ", time.time()-a

        a = time.time()
        ind_ptr = [0]
        data = []
        indices = []
        for i in range(A):
            data.append(1)
            indices.append(active_outputs[i])
            ind_ptr.append(len(data))
        trans = scipy.sparse.csr_matrix((data, indices, ind_ptr),
                                        shape=(A, I),
                                        dtype=np.uint8)
        # print  "SPARSE time ", time.time()-a
        #D x A  A x I  I x N
        return (features.T * trans * mat).T


        # a = time.time()
        # # print len(output_set)
        # data = []
        # indices = []
        # ind_ptr = [0]
        # d2 = {}
        # for n in range(N):
        #     feats = []
        #     for index in mat[:, n].nonzero():
        #         if not index: continue
        #         data.append(1)
        #         if index[0] in d2:
        #             indices.append(d2[index[0]])
        #         else:
        #             d2[index[0]] = len(feats)
        #             indices.append(d2[index[0]])
        #             element = output_set.element(index)
        #             features.append(self.element_features(element, templates, preprocessed_x))
        #     ind_ptr.append(len(data))

        # f_mat = features
        # print  "INNER time ", time.time()-a
        # trans = scipy.sparse.csc_matrix((data,indices,ind_ptr),
        #                                 shape=(len(features), N),
        #                                 dtype=np.uint8)

        # a = time.time()
        # mat = self.encoder.transform(f_mat)
        # assert mat.shape == (len(features), D)


        # edge_features = trans.T * mat
        # print  "TRANS time ", time.time()-a
        # # a = time.time()
        # #mat = scipy.sparse.vstack(features)
        # # print  "STACK time ", time.time()-a
        # assert edge_features.shape == (N, D)
        # return edge_features

    def inference(self, x, w, relaxed=False):
        """
        Parameters
        -----------
        x :

        w : D x 1 matrix

        """
        a = time.time()
        chart = self.chart(x)

        hypergraph = chart.finish()

        # print "CHART time",  time.time() - a, len(hypergraph.edges)

        a = time.time()
        I = len(chart.output_set)
        N = len(hypergraph.edges)
        D = self.encoder.feature_indices_[-1]

        assert w.shape == (D,)

        a = time.time()
        feature_matrix = self._feature_matrix(x, chart.matrix(), chart.output_set)
        assert feature_matrix.shape == (N, D)
        # print "FEATURE MATRIX time",  time.time() - a

        # a = time.time()
        mat = chart.matrix()
        assert mat.shape == (I, N)

        theta = w.T * feature_matrix.T
        assert theta.shape == (N,)


        a = time.time()

        path = pydecode.best_path(hypergraph, theta)

        # I x 1 matrix
        output = mat * path.v
        assert output.shape == (I, 1)

        # for n in range(N):
        #     for index in mat[:, n].nonzero():
        #         if not index: continue
        #         element = chart.output_set.element(index[0])
        #         if theta[n] != 0:
        #             print element, theta[n]

        element_output = [chart.output_set.element(j)
                          for j in output.nonzero()[0]]
        #print element_output
        # print "OUTPUT time",  time.time() - a

        out = self._output_coder.inverse_transform(element_output)
        return out



    # def joint_feature_initialize(self, x, y):
    #     templates = self.feature_templates()
    #     output_set = self.output_set(x)
    #     preprocessed_x = self
    #     return [self.element_features(output_set.element(index), templates, preprocessed_x)
    #             for index in y.nonzero()[0]]

    def joint_feature(self, x, y):
        """
        Returns
        --------
        Features : I x D matrix
        """
        elements = self._output_coder.transform(y)

        D = self.encoder.feature_indices_[-1]

        templates = self._wrapped_feature_templates()
        #output_set = self.output_set(x)
        preprocessed_x = self._preprocessor.preprocess(x)

        cat_feats = [self.element_features(element,
                                         templates, preprocessed_x)
                    for element in elements]
        features = self.encoder.transform(cat_feats)

        # Sum up the features.
        final_features = features.T * np.ones(features.shape[0]).T
        assert final_features.shape == (D, )
        return final_features

    def loss(self, yhat, y):
        pass
