
PyDecode is a dynamic programming toolkit for researchers studying natural language processing.

The interface and visualization code is written Python, the core algorithms are written in C++.
For a full description of the project see the documentation_.

.. _documentation: http://pydecode.readthedocs.org/


.. image:: _images/parsing_9_0.png
   :width: 500 px
   :align: center




Benefits
-------------

* **Simple imperative interface.** Algorithms look like pseudo-code ::

    def viterbi(chart, words):
        c.init(Tagged(0, "ROOT"))
        for i, word in enumerate(words[1:], 1):
            for tag in emission[word]:
                c[Tagged(i, tag)] = \
                   c.sum((c[Tagged(i - 1, prev)] * \
                          c.sr(Bigram(word, tag, prev))
                          for prev in emission[words[i-1]]))

* **Efficient implementation.** Python front-end constructs C++ data structures.


  * If you need even more efficiency, you can use the hypergraph interface directly.



* **Easy-to-use extensions.** Write only the max dynamic program.

  * PyDecode provides the derivations, posteriors, max-marginals, and oracle scoring.

  * Add some features and it can do structured training.

  * Add constraints and it can run Lagrangian relaxation.


Documentation, Tutorial and Gallery
----------------------

.. hlist::
   :columns: 2

   * documentation_
   * tutorial_
   * gallery_
   * api_


Features
-------------

Currently the toolkit is in early development but should be ready to be used.
This presentation_ discusses the background for this work.

.. _presentation: https://github.com/srush/PyDecode/raw/master/writing/slides/slides.pdf

Dynamic Programming
======================

* Simple imperative construction of dynamic programming structures.
* Customizable GraphViz output for debugging.
* Algorithms for best path, inside scores, outside scores, and oracle scores.
* Pruning based on max-marginals.
* Semiring operations over hypergraph structures.
* Integration with an LP solver.

Constrained Dynamic Programming
===============================

* Hypergraphs with additional constraints.
* Lagrangian Relaxation optimization of constrained problems.
* ILP optimization optimization of constrained problems.

Structured Prediction
===============================

* Hooks into PyStruct for structured training.

Coming Soon
===============================

* A*, beam search, coarse-to-fine extensions.
* Faster K-best algorithms.


.. image:: https://travis-ci.org/srush/PyDecode.png?branch=master
    :target: https://travis-ci.org/srush/PyDecode

.. _gallery: http://pydecode.readthedocs.org/en/latest/notebooks/gallery.html
.. _tutorial: http://pydecode.readthedocs.org/en/latest/notebooks/index.html
.. _api: http://pydecode.readthedocs.org/en/latest/api.html
