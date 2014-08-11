
pydecode.StructuredEncoder
==========================


.. currentmodule:: pydecode.nlp
.. autofunction:: StructuredEncoder

Examples
--------


.. code:: python

    import pydecode
    import pydecode.encoder
    import numpy as np
.. code:: python

    tags = ["D", "V", "N", "A"]
    sentence = "the dog walked to the park".split()
.. code:: python

    class TaggingEncoder(pydecode.encoder.StructuredEncoder):
        def __init__(self, tags, sentence):
            self.T = len(tags)
            self.n = len(sentence)
            shape = (self.n, self.T)
            super(TaggingEncoder, self).__init__(shape)
    
        def from_parts(self, parts):
            tag_sequence = np.zeros(self.n)
            for part in parts:
                tag_sequence[part[0]] = part[1]
            return tag_sequence
    
        def transform_structure(self, structure):
            parts = []
            for i, t in enumerate(structure):
                parts.append((i,t)) 
            return np.array(parts)
    
    encoder = TaggingEncoder(tags, sentence)
.. code:: python

    tag_sequence = np.array([3,2,3,1,0, 2])
    parts = encoder.transform_structure(tag_sequence)
    parts



.. parsed-literal::

    array([[0, 3],
           [1, 2],
           [2, 3],
           [3, 1],
           [4, 0],
           [5, 2]])



.. code:: python

    encoder.from_parts(parts)



.. parsed-literal::

    array([ 3.,  2.,  3.,  1.,  0.,  2.])



.. code:: python

    labels = encoder.encoder[tuple(parts.T)]
    labels



.. parsed-literal::

    array([ 3,  6, 11, 13, 16, 22])



.. code:: python

    parts = encoder.transform_labels(labels)
    parts



.. parsed-literal::

    array([[0, 3],
           [1, 2],
           [2, 3],
           [3, 1],
           [4, 0],
           [5, 2]])



Invariants
----------


Transform between parts and labels and parts is identity.

.. code:: python

    def test_transform():
        shape = (10, 15)
        encoder = pydecode.encoder.StructuredEncoder(shape)
        a = np.random.randint(10, size=10)
        b = np.random.randint(15, size=10)
    
        parts = np.vstack((a.T, b.T)).T
        labels = encoder.encoder[tuple(parts.T)]
        reparts = encoder.transform_labels(labels)
        assert (parts == reparts).all()
    test_transform()