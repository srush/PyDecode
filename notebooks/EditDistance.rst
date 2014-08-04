
Tutorial 2: Edit Distance
=========================


In this tutorial, we look at a more substantial dynamic programming
problem: computin the edit distance between two strings. In this
notebook, we show how to

-  Construct a dynamic program.
-  Assign scores to outputs.
-  Use more advanced hypergraph operations.

We beging by importing the standard libraries.

.. code:: python

    import pydecode
    import numpy as np
    import matplotlib.pyplot as plt
Edit distance problems are concerned with aligning two strings. An alignment consists of a 
sequence of single operations 

* Match : Generate the same character in both string.
* Insert : Generate a character only in the first string.
* Delete : Generate a character only in the second string.

For a detailed overview of the edit distance problem, see the wikipedia article on `Levenshtein distance <http://en.wikipedia.org/wiki/Levenshtein_distance>`_.,

.. _`Levenshtein distance`  http://en.wikipedia.org/wiki/Levenshtein_distance.

We begin by defining the basic operations available for edit distance,
and giving the dynamic program for edit distance.

.. code:: python

    kInsert, kDelete, kMatch = 0, 1, 2
    operations = np.array([kInsert, kDelete, kMatch])
    offsets = np.array([[1,0], [0,1], [1,1]]) 
    op_names = np.array(["<",  ">",  "="])
.. code:: python

    def grid(m, n, o=None):
        if o is None:
            return np.arange(m * n).reshape((m, n))
        else:
            return np.arange(m * n * o).reshape((m, n, o))
    grid(5,5)



.. parsed-literal::

    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]])




Edit distance is defined by the following base case and recursions 

.. math::
    C_{0, 0} &=& 0\ \ \  \forall \  j \in \{ 0 \ldots m \} \\
    C_{0, j} &=& 0\ \ \ \forall \  j \in \{ 1 \ldots m \} \\
    C_{i, 0} &=& 0\ \ \  \forall\  i \in \{ 1 \ldots n \} \\
    \\
    C_{i, j} &=& \max \begin{array}{ll}C_{i-1, j} +  W_{i-1, j, DELETE} \\ C_{i, j-1}  + W_{i, j-1, INSERT} \\ C_{i-1, j-1} + W_{i-1, j-1, MATCH}\end{array}    \\


.. code:: python

    def edit_distance(m, n):
    
        # Create a grid for the items and labels.
        item_grid = grid(m, n)
        op_grid = grid(m, n, len(operations))
        c = pydecode.ChartBuilder(item_grid)
    
        # Construct the base cases. 
        c.init(item_grid[0, 0])
        c.init(item_grid[0, 1:])
        c.init(item_grid[1:, 0])
    
        for i in range(1, m):
            for j in range(1, n):
                position = np.array([i, j])
                prev = []
                labels = []
                for op, offset in zip(operations, offsets):
                    prev_i, prev_j = position - offset
                    prev.append([item_grid[prev_i, prev_j]])
                    labels.append(op_grid[prev_i, prev_j, op])
                c.set(item_grid[i, j], prev, labels)
        return c.finish()
.. code:: python

    graph = edit_distance(3, 3)
Let's look at the the dynamic program in more detail.

First note that we have used an indexing trick from numpy to define out
chart. The set of chart items is now more interesting than before. We
have one item for each string1 x string2 position. Offsets are used to
calculate the previous element of the chart.

.. code:: python

    items



.. parsed-literal::

    array([[ 0,  1,  2,  3,  4,  5],
           [ 6,  7,  8,  9, 10, 11],
           [12, 13, 14, 15, 16, 17],
           [18, 19, 20, 21, 22, 23],
           [24, 25, 26, 27, 28, 29],
           [30, 31, 32, 33, 34, 35]])



.. code:: python

    We also have a output set which describes the operations applied at each point in the dynamic program.
    The key part of the function is the call to ``set``.
c.set(items[i,j],
=================

items[prev\_pos[:,0], prev\_pos[:,1]],
======================================

out=outputs[prev\_pos[:,0], prev\_pos[:,1], operations])
========================================================

This indicates that item (i, j) should be constructed from the array of previous items each associated with
===========================================================================================================

an output structure from ``outputs``. These two arrays must be of the same size.
================================================================================


To get a better sense of this dynamic program, we can look at its
hypergraph.

.. code:: python

    # Construct readable labels for each of the vertices and edges in the graph.
    vertex_labels = ["%s | %s"%(strings[0][a-1], strings[1][b-1])
                     for a, b in pydecode.vertex_items(dp)]
    hyperedge_labels = op_names[pydecode.hyperedge_outputs(dp)[2]]
    display.HypergraphFormatter(dp.hypergraph, vertex_labels=vertex_labels, hyperedge_labels=hyperedge_labels).to_ipython()



.. image:: EditDistance_files/EditDistance_15_0.png



This structure can then be used for queries about the underlying
strings. First, we might ask what the best alignment is between the two
strings. To do this, we need to assign as score to each output in the
dynamic program. Each of these outputs corresponds to choosing an
operation at each of the position pair.

Let's give all operations a score of zero, except for Match which can
only be applied when we have a direct match.

.. code:: python

    def make_scores(strings, outputs):
        output_scores = np.zeros(outputs.shape)
        for i, s in enumerate(strings[0], 1):
            for j, t in enumerate(strings[1], 1):
                output_scores[i, j, kMatch] = 1.0 if s == t else -1e8
        return output_scores
    output_scores = make_scores(strings, dp.outputs)
Finding the best alignment is simply a matter of calling the argmax
function. This retuns the best outputs under our scoring function. We
can then transform these into an easier to view format.

.. code:: python

    best = pydecode.argmax(dp, output_scores)
    best



.. parsed-literal::

    array([[4, 4, 2],
           [3, 3, 2],
           [2, 3, 0],
           [2, 2, 1],
           [1, 1, 2]])



.. code:: python

    chart = np.zeros(dp.outputs.shape[:2])
    chart[best.T[:2][0], best.T[:2][1]] = 1
    plt.pcolor(chart)
    plt.yticks(np.arange(1.5, len(strings[0])+1, 1), strings[0])
    plt.xticks(np.arange(1.5, len(strings[1])+1, 1), strings[1])
    None


.. image:: EditDistance_files/EditDistance_20_0.png


Furthermore, we can map these scores directly onto the hypergraph, to
see which path was chosen as the highest scoring.

.. code:: python

    hypergraph_scores = dp.output_matrix.T * output_scores.ravel()
    path = pydecode.best_path(dp.hypergraph, hypergraph_scores)
    display.HypergraphPathFormatter(dp.hypergraph, vertex_labels=vertex_labels, hyperedge_labels=hyperedge_labels).set_paths([path]).to_ipython()



.. image:: EditDistance_files/EditDistance_22_0.png



Another common query is for the max-marginals of a given dynamic
program. The max-marginals given the highest scoring alignment that uses
a particular item or output in the dynamic program. These can be very
useful for pruning, training models, and decoding with partial data.

.. code:: python

    output_marg = pydecode.output_marginals(dp, output_scores)
    output_marg[2, 3, :]



.. parsed-literal::

    array([  3.00000000e+00,   2.00000000e+00,  -9.99999980e+07])



.. code:: python

    plt.imshow(output_marg[:,:])
    plt.yticks(np.arange(1.5, len(strings[0])+1, 1), strings[0])
    plt.xticks(np.arange(1.5, len(strings[1])+1, 1), strings[1])
    None



.. image:: EditDistance_files/EditDistance_25_0.png


Finally we look at a longer alignment example.

.. code:: python

    strings = np.array(["hllo this is a longer sequence", 
                        "hello ths is a longr seqence"])
    dp = edit_distance(strings)
    output_scores = make_scores(strings, dp.outputs)
    best = pydecode.argmax(dp, output_scores)
.. code:: python

    chart = np.zeros(dp.items.shape)
    chart[best.T[:2][0], best.T[:2][1]] = 1
.. code:: python

    plt.imshow(chart)
    plt.yticks(np.arange(1.5, len(strings[0])+1, 1), strings[0])
    plt.xticks(np.arange(1.5, len(strings[1])+1, 1), strings[1])
    None


.. image:: EditDistance_files/EditDistance_29_0.png

