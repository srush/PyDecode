.. toctree::
   :maxdepth: 2

==========
Dynamic Programming
==========


Bottom-Up Dynamic Programming
=============================

In computer science, there are several ways of viewing dynamic
programming algorithms. Two of the most common are the top-down
(recursive) view and the bottom-up (chart) view. The PyDecode API utilizes
the bottom-up view to specify dynamic programming algorithms.

From a bottom-up perspective, a dynamic program begins with a set of 
initial base cases, and then consists of a series
of calculations assigning a value to an `item` based on a combination
of previously calculated `item` values and a set of `scores`.

Formally, define the set of items as :math:`{\cal I}` and the chart as a vector
in :math:`{\mathbb R}^{|{\cal I}|}`. Initialization consists of assigning a null value 
to a 

:math:`C_i \gets 0`

Each subsequent stage :math:`1 \ldots K` of the dynamic program consists of altering an internal value ::

:math:`C_i \gets C_i \oplus (C_(j_1) \otimes \ldots \otimes C_{j_n} \otimes s(\rho_k))`

where :math:`C` is the chart vector, :math:`i`: is the current item, and
:math:`j_1\ldots j_n` are previously calculated items. Additionally 
the stage `label` is :math:`\rho_k` which comes from the set :math:`{\cal L}`.
The label is assigned a `score` based on a scoring function `s`.

Each item may be assigned a value in several different stages. Next consider the 
value the sum of each of these assignments. 

:math:`C_i \gets (C_(j_{1,1}) \otimes \ldots \otimes C_{j_{1,n}} \otimes s(\rho_k)) \oplus \\ 
                  (C_(j_{2,1}) \otimes \ldots \otimes C_{j_{2,n}} \otimes s(\rho_{k+1})) \oplus \\ 
                  
                  (C_(j_{m,1}) \otimes \ldots \otimes C_{j_{m,n}} \otimes s(\rho_{k + n -1}))  \\`

We can think of this as combining several vectors down.

:math:`C_i \gets \Oplus (C_(j_{*,1}) \otimes \ldots \otimes C_{j_{*,n}} \otimes s(\rho_{k..k+n-1}))


Construction
=============================

Next let's consider how this is done in PyDecode. First we construct the sets
for items and labels respectively ::

  import numpy as np
  items = np.arange().reshape()
  labels = np.arange().reshape()
  chart = ChartBuilder(items, labels)
  chart.init(items[1,1])
  
Next we specify the main recursion of the dynamic program ::

  for i in items:
      j = previous_items(i)
      rho = labels(i)
      chart.set(i, j[:, 1], j[:, 2], j[:, 3], out=rho) 


Finally we finish and get a DynamicProgram object:: 

  dp = chart.finish() 

.. _dp:


Construction
===================

.. automodule:: pydecode.potentials
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/
   :template: class.rst

   ChartBuilder
   DynamicProgram

.. _hypergraph:


Algorithms
==========

.. automodule:: pydecode
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/

   argmax
   fill
   marginals
   score_outputs


Semirings
============

Each of these algorithms is parameterized over several
different semirings. The `kind` argument is used to specify
the semiring.

.. automodule:: pydecode.potentials
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/
   :template: class.rst

   LogViterbi
   Inside
   Bool


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
