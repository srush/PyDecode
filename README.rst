
PyDecode is a dynamic programming toolkit developed for research in natural langauge processing. Its aim is to be simple enough for fast prototyping, but efficient enough for research use.


.. _documentation: http://pydecode.readthedocs.org/


.. image:: _images/parsing_9_0.png
   :width: 500 px
   :align: center





* **Simple specifications.** Dynamic programming algorithms specified through pseudo-code. ::

    ...
    c.init(items[0, K[0]])
    for i in range(1, n):
        for t in K[i]:
            c.set(items[i, t],
                  items[i-1, K[i-1]],
                  out=outputs[i, t, K[i-1]])
    dp = c.finish()

* **Efficient implementation.** Core code in C++, interfaces through Numpy. ::

    scores = numpy.random.random(dp.outputs.shape)
    y = pydecode.argmax(dp, scores)

* **High-level algorithms.** Includes a set of widely-used algorithms. ::

    # Max-marginals.
    marginals = pydecode.output_marginals(dp, scores)

    # Inside probabilities.
    inside = pydecode.fill(dp, scores, kind=pydecode.Inside)

    # Pruning
    keep = values > threshold
    _, projection, pruned = pydecode.project(dp.hypergraph, keep)

* **Integration with machine learning toolkits.** Train structured models. ::


    perceptron_tagger = StructuredPerceptron(tagger)
    perceptron_tagger.fit(X, Y)
    Y_test = perceptron_tagger.predict(X_test)

* **Visualization tools.**  IPython integrated tools for debugging algorithms. ::

    display.HypergraphFormatter(hypergraph).to_ipython()

.. image:: _images/hmm.png
   :width: 500 px
   :align: center


.. Documentation, Tutorial and Gallery
.. ----------------------

.. .. hlist::
..    :columns: 2

..    * documentation_
..    * tutorial_
..    * gallery_
..    * api_


.. Features
.. -------------

.. Currently the toolkit is in development. It includes the following features:

.. * Simple construction of dynamic programs.
.. * Customizable GraphViz output for debugging.
.. * Algorithms for best path, inside scores, outside scores, and oracle scores.
.. * Several types of pruning.
.. * Integration with an (I)LP solver for constrained problems.
.. * Lagrangian Relaxation optimization tools.
.. * Semiring operations over hypergraph structures.
.. * Hooks into PyStruct for structured training.
.. * Fast k-best algorithms.


.. .. image:: https://travis-ci.org/srush/PyDecode.png?branch=master
..     :target: https://travis-ci.org/srush/PyDecode

.. _gallery: http://pydecode.readthedocs.org/en/latest/notebooks/index.html
.. _tutorial: http://pydecode.readthedocs.org/en/latest/notebooks/index.html
.. _api: http://pydecode.readthedocs.org/en/latest/api.html
