
pydecode.nlp.cfg
================


.. currentmodule:: pydecode.nlp
.. autofunction:: cfg

Examples
--------


.. code:: python

    import pydecode, pydecode.nlp, pydecode.test
    import numpy as np
Bibliography
------------




Invariants
----------


.. code:: python

    def test_all_paths(sentence_length, grammar_size):
        graph, encoder = pydecode.nlp.cfg(sentence_length, 
                                          grammar_size)
    
        # Generate all paths.
        p1 = np.array([encoder.transform_path(path).ravel()
                       for path in pydecode.test.all_paths(graph)])
    
        # Generate all parses.
        p2 = np.array([parse.ravel()
                       for parse in encoder.all_structures()])
        assert (p1[np.lexsort(p1.T)] == p2[np.lexsort(p2.T)]).all()
    
    for length in range(3, 6):
        for grammar in range(1, 3):
            test_all_paths(length, grammar)