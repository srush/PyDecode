import pystruct.StructuredModel
import pydecode.hyper as ph
import numpy as np

class HypergraphModelBuilder:
    def features(self, obj):
        pass



class HypergraphModel(StructuredModel):
    """

    """
    def __init__(self, hypergraph_builder):
        self.hypergraph_builder = hypergraph_builder

    def _build_hypergraph(self):
        hypergraph = ph.Hypergraph()
        hypergraph.build(self.hypergraph_builder.build)
        return hypergraph

    def _build_weights(self, w):
        def weight_builder(label):
            return w * self.hypergraph_builder.features(label)
        weights = ph.Weights(hypergraph)
        weights.build(weight_builder)
        return weights

    def _path_features(self, hypergraph, path):
        features = np.array() 
        for edge in path:
            f = self.hypergraph_builder.features(hypergraph.label(edge))
            features += f
        return features 

    def inference(self, x, w):
        hypergraph = self._build_hypergraph()
        weights = self._build_weights(w)
        path, _ = ph.best_path(hypergraph, weights)
        return self._path_features(hypergraph, path)
        
    def loss(self, yhat, y):
        pass

    
