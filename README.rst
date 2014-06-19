
PyDecode is a dynamic programming toolkit, developed for research in  NLP.

The aim is to be simple enough for prototyping, but efficient enough for research use.


.. _documentation: http://pydecode.readthedocs.org/


.. image:: _images/parsing_9_0.png
   :width: 500 px
   :align: center


Features
-------------

* **Simple interface.** Dynamic programming algorithms specified through pseudo-code. ::

    c[0, START] = c.init()

    for i in range(1, n):
        for tag in tags:
            c[i, tag] = \
                [c.merge((i-1, prev), values=[(i-1, tag, prev)])
                 for prev in tags]

* **Efficient implementation.** Core code in C++, interfaces through numpy/scipy. ::

    dp = c.finish()
    scores = numpy.random(len(dp.edges))
    path = pydecode.best_path(dp, scores)
    scores = scores.T * path.v

* **High-level algorithms.** Includes a set of widely-used algorithms. ::

    # Max-marginals.
    marginals = pydecode.marginals(dp, scores)

    # K-Best decoding.
    kbest = pydecode.kbest(dp, scores)

    # Inside probabilities.
    inside = pydecode.inside(dp, scores, kind=pydecode.Inside)

    # Pruning
    filter = marginals > threshold
    _, projection, pruned_dp = pydecode.project(dp, filter)

* **Integration with machine learning toolkits.** Train structured models using dynamic programming. ::


    perceptron_tagger = StructuredPerceptron(tagger, max_iter=5)
    perceptron_tagger.fit(X, Y)
    Y_test = perceptron_tagger.predict(X_test)

* **Visualization tools.**  IPython integrated tools for debugging algorithms. ::

    display.HypergraphFormatter(labeled_dp).to_ipython()


.. image:: _images/hmm.png
   :width: 500 px
   :align: center


Documentation, Tutorial and Gallery
----------------------

.. hlist::
   :columns: 2

   * documentation_
   * tutorial_
   * gallery_
   * api_


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


.. image:: https://travis-ci.org/srush/PyDecode.png?branch=master
    :target: https://travis-ci.org/srush/PyDecode

.. _gallery: http://pydecode.readthedocs.org/en/latest/notebooks/index.html
.. _tutorial: http://pydecode.readthedocs.org/en/latest/notebooks/index.html
.. _api: http://pydecode.readthedocs.org/en/latest/api.html
