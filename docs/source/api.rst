
==========
API
==========



.. _dp:

Construction
===================

Chart

.. automodule:: pydecode._pydecode
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/
   :template: class.rst

   ChartBuilder


Hypergraph

.. automodule:: pydecode
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Hypergraph
   Edge
   Vertex
   Path




.. _hypergraph:

Algorithms
==========

.. .. automodule:: pydecode
..    :no-members:
..    :no-inherited-members:

.. .. autosummary::
..    :toctree: generated/

Table



==============  =========  =====================================================
Algorithm                                              Description
==============  =========  =====================================================
**best_path**   |best|     Find the highest-weight hyperpath.
**inside**      |ins|      Compute the inside weight table.
**outside**     |outs|     Compute the outside weight table.
**marginals**   |marg|     Compute the hyperedge marginals.
**transform**   |trans|    Convert between label and hyperedge representation.
**binarize**    |bin|      Convert to a binary-branching hypergraph.
**kbest**       |kbest|    Find the k-highest scoring hyperpaths.
**intersect**   |inter|    Intersect with a finite-state acceptor.
**draw**        |draw|     Visualize the hypergraph.
**lp**          |lp|       Build linear program.
==============  =========  =====================================================

.. |best| replace:: [:doc:`doc<notebooks/doc/best_path>`]
.. |ins| replace:: [:doc:`doc<notebooks/doc/inside>`]
.. |outs| replace:: [:doc:`doc<notebooks/doc/outside>`]
.. |marg| replace:: [:doc:`doc<notebooks/doc/marginals>`]
.. |trans| replace:: [:doc:`doc<notebooks/doc/transform>`]
.. |bin| replace:: [:doc:`doc<notebooks/doc/binarize>`]
.. |kbest| replace:: [:doc:`doc<notebooks/doc/kbest>`]
.. |inter| replace:: [:doc:`doc<notebooks/doc/intersect>`]
.. |draw| replace:: [:doc:`doc<notebooks/doc/draw>`]
.. |lp| replace:: [:doc:`doc<notebooks/doc/lp>`]

.. _weight_types:

Weight Types
============

Each of these algorithms is parameterized over several
different semirings. The ``weight_type`` argument is used to specify
the semiring.

==============  ==============  ===============  ===============  ===============  =======
Name            :math:`\oplus`  :math:`\otimes`  :math:`\bar{0}`  :math:`\bar{1}`  dtype
==============  ==============  ===============  ===============  ===============  =======
**LogViterbi**   :math:`\max`    :math:`+`       :math:`-\infty`  0                float32
**Viterbi**      :math:`\max`    :math:`*`       0                1                float32
**Real**         :math:`+`       :math:`*`       0                1                float32
**Log**          logsum          :math:`+`       :math:`-\infty`   0               float32
**Boolean**      or               and             false             true            uint8
**Counting**     :math:`+`       :math:`*`        0                 1              int32
**MinMax**       :math:`\min`    :math:`\max`    :math:`-\infty`  :math:`\infty`   float32
==============  ==============  ===============  ===============  ===============  =======




.. =====  =====  =======
.. A      B      A and B
.. =====  =====  =======
.. False  False  False
.. True   False  False
.. False  True   False
.. True   True   True
.. =====  =====  =======

.. .. automodule:: pydecode.potentials
..    :no-members:
..    :no-inherited-members:

.. .. autosummary::
..    :toctree: generated/
..    :template: class.rst

..    LogViterbi
..    Inside
..    Bool


.. Potentials
.. ============

.. Users can specify potential vectors that are associated with
.. each edge of the hypergraph.

.. .. automodule:: pydecode.potentials
..    :no-members:
..    :no-inherited-members:

.. .. autosummary::
..    :toctree: generated/
..    :template: class.rst

..    Potentials

.. There are several types of potentials implemented.

.. .. automodule:: pydecode.potentials
..    :no-members:
..    :no-inherited-members:

.. .. autosummary::
..    :toctree: generated/
..    :template: class.rst

..    LogViterbiPotentials
..    InsidePotentials
..    BoolPotentials


.. Using the corresponding data structures


.. .. automodule:: pydecode.potentials
..    :no-members:
..    :no-inherited-members:

.. .. autosummary::
..    :toctree: generated/

..    Chart
..    Marginals
..    BackPointers

.. Graph Builder
.. ------------

.. .. automodule:: pydecode.potentials
..    :no-members:
..    :no-inherited-members:

.. .. autosummary::
..    :toctree: generated/
..    :template: class.rst

..    GraphBuilder
