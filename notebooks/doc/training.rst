
pydecode.model.DynamicProgrammingModel
======================================


.. currentmodule:: pydecode.nlp
.. autofunction:: StructuredEncoder

Examples
--------


.. code:: python

    import pydecode
    import pydecode.nlp
    from pydecode.model import DynamicProgrammingModel, HammingLossModel
    from pystruct.learners import StructuredPerceptron
    import numpy as np
.. code:: python

    tags = ["D", "V", "N", "P"]
    n_tags = len(tags)
    n_words = 10
.. code:: python

    class SimpleTagModel(HammingLossModel, DynamicProgrammingModel):
        def templates(self): 
            return [(n_tags, n_tags),
                    (n_tags, n_words),
                    (n_tags, n_tags, n_words)]
        
        def parts_features(self, x, parts): 
            x_arr = np.array(x)
            return [(parts[:,1], parts[:,2]),
                    (parts[:,1], x_arr[parts[:,0]]),
                    (parts[:,1], parts[:,2], x_arr[parts[:,0]])]
    
        def dynamic_program(self, x): 
            n = len(x)
            return pydecode.nlp.tagger(n, [1]+[len(tags)] * (n-2) +[1])
.. code:: python

    model = SimpleTagModel()
    sp = StructuredPerceptron(model, verbose=True, max_iter=10, average=True)
.. code:: python

    X = [(0,1,2,3, 0), (0,2,2,3,0)]
    Y = [(0,1,2,3, 0), (0,2,2,3,0)]
    sp.fit(X, Y)
    None

.. parsed-literal::

    iteration 0
    avg loss: 0.400000 w: [[-1.]
     [ 0.]
     ..., 
     [ 0.]
     [ 0.]]
    effective learning rate: 1.000000
    iteration 1
    avg loss: 0.200000 w: [[-1.]
     [ 0.]
     ..., 
     [ 0.]
     [ 0.]]
    effective learning rate: 1.000000
    iteration 2
    avg loss: 0.200000 w: [[-1.]
     [ 0.]
     ..., 
     [ 0.]
     [ 0.]]
    effective learning rate: 1.000000
    iteration 3
    avg loss: 0.000000 w: [[-1.]
     [ 0.]
     ..., 
     [ 0.]
     [ 0.]]
    effective learning rate: 1.000000
    Loss zero. Stopping.


.. code:: python

    sp.predict(X)



.. parsed-literal::

    [array([0, 1, 2, 3, 0], dtype=int32), array([0, 2, 2, 3, 0], dtype=int32)]



Bibliography
------------


.. bibliography:: ../../full.bib 
   :filter: key in {"collins02perc"}
   :style: plain

Invariants
----------


.. code:: python

    