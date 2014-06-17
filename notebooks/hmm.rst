
.. code:: python

    import pydecode.hyper as ph
    import pydecode.display as display
    import pandas as pd
    import matplotlib.pyplot as plt
Tutorial 3: HMM Tagger
======================


We begin by constructing the HMM probabilities.

.. code:: python

    # The emission probabilities.
    
    tags = ["ROOT", "D", "N", "V", "END"]
    
    emission = {'ROOT' : {'ROOT' : 1.0},
                'the' :  {'D': 0.8, 'N': 0.1, 'V': 0.1},
                'dog' :  {'D': 0.1, 'N': 0.8, 'V': 0.1},
                'walked':{'V': 1},
                'in' :   {'D': 1},
                'park' : {'N': 0.1, 'V': 0.9},
                'END' :  {'END' : 1.0}}
    
    # The transition probabilities.
    transition = {'D' :    {'D' : 0.1, 'N' : 0.8, 'V' : 0.1, 'END' : 0},
                  'N' :    {'D' : 0.1, 'N' : 0.1, 'V' : 0.6, 'END' : 0.2},
                  'V' :    {'D' : 0.4, 'N' : 0.3, 'V' : 0.2, 'END' : 0.1},
                  'ROOT' : {'D' : 0.4, 'N' : 0.3, 'V' : 0.3},
                  'END': {'END' : 1.0}}
.. code:: python

    pd.DataFrame(transition).fillna(0) 



.. raw:: html

    <div style="max-height:1000px;max-width:1500px;overflow:auto;">
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>D</th>
          <th>END</th>
          <th>N</th>
          <th>ROOT</th>
          <th>V</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>D</th>
          <td> 0.1</td>
          <td> 0</td>
          <td> 0.1</td>
          <td> 0.4</td>
          <td> 0.4</td>
        </tr>
        <tr>
          <th>END</th>
          <td> 0.0</td>
          <td> 1</td>
          <td> 0.2</td>
          <td> 0.0</td>
          <td> 0.1</td>
        </tr>
        <tr>
          <th>N</th>
          <td> 0.8</td>
          <td> 0</td>
          <td> 0.1</td>
          <td> 0.3</td>
          <td> 0.3</td>
        </tr>
        <tr>
          <th>V</th>
          <td> 0.1</td>
          <td> 0</td>
          <td> 0.6</td>
          <td> 0.3</td>
          <td> 0.2</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: python

    pd.DataFrame(emission).fillna(0)



.. raw:: html

    <div style="max-height:1000px;max-width:1500px;overflow:auto;">
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>END</th>
          <th>ROOT</th>
          <th>dog</th>
          <th>in</th>
          <th>park</th>
          <th>the</th>
          <th>walked</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>D</th>
          <td> 0</td>
          <td> 0</td>
          <td> 0.1</td>
          <td> 1</td>
          <td> 0.0</td>
          <td> 0.8</td>
          <td> 0</td>
        </tr>
        <tr>
          <th>END</th>
          <td> 1</td>
          <td> 0</td>
          <td> 0.0</td>
          <td> 0</td>
          <td> 0.0</td>
          <td> 0.0</td>
          <td> 0</td>
        </tr>
        <tr>
          <th>N</th>
          <td> 0</td>
          <td> 0</td>
          <td> 0.8</td>
          <td> 0</td>
          <td> 0.1</td>
          <td> 0.1</td>
          <td> 0</td>
        </tr>
        <tr>
          <th>ROOT</th>
          <td> 0</td>
          <td> 1</td>
          <td> 0.0</td>
          <td> 0</td>
          <td> 0.0</td>
          <td> 0.0</td>
          <td> 0</td>
        </tr>
        <tr>
          <th>V</th>
          <td> 0</td>
          <td> 0</td>
          <td> 0.1</td>
          <td> 0</td>
          <td> 0.9</td>
          <td> 0.1</td>
          <td> 1</td>
        </tr>
      </tbody>
    </table>
    </div>



Next we specify the labels for the transitions.

And the scoring function.

.. code:: python

    def item_set(n):
        return ph.IndexSet((n, len(tags)))
    
    def output_set(n):
        return ph.IndexSet((n, len(tags), len(tags)))
.. code:: python

    def scores(words):
        n = len(words)
        outputs = output_set(n)
        scores = np.zeros(len(outputs))
        for j, (i, tag, prev_tag) in outputs.iter_items():
            scores[j] = transition[tags[prev_tag]].get(tags[tag], 0.0) * \
                emission[words[i]].get(tags[tag], 0.0)
        return scores
.. code:: python

    def viterbi(n):
        c = ph.ChartBuilder(item_set=item_set(n), 
                            output_set=output_set(n))
        for tag in range(len(tags)):
            c[0, tag] = c.init()
        for i in range(1, n-1):
            for tag in range(len(tags)):
                c[i, tag] = \
                    [c.merge((i-1, prev), values=[(i, tag, prev)])
                     for prev in range(len(tags))]
    
        c[n-1, 0] = [c.merge((n-2, prev), values=[(n-1, len(tags)-1, prev)]) 
                     for prev in range(len(tags))]
        return c
Now we write out dynamic program.

Now we are ready to build the structure itself.

.. code:: python

    # The sentence to be tagged.
    sentence = 'ROOT the dog walked in the park END'.split()
.. code:: python

    score_vector = scores(sentence)
.. code:: python

    chart = viterbi(len(sentence))
    hypergraph = chart.finish()
    outputs = chart.matrix()
    item_mat = chart.item_matrix()
    output_ = output_set(len(sentence))
    items = item_set(len(sentence))
.. code:: python

    theta = score_vector * outputs
    path = ph.best_path(hypergraph, theta, kind=ph.Inside)
But even better we can construct the entrire search space. We can also
output the path itself. We can also use a custom fancier formatter.
These attributes are from graphviz
(http://www.graphviz.org/content/attrs)

.. code:: python

    node_marg, edge_marg = ph.marginals(hypergraph, theta, kind=ph.Inside)
    normalized_marg = node_marg / node_marg[hypergraph.root.id]
.. code:: python

    m = min(normalized_marg)
    M = max(normalized_marg)
    
    class HMMFormat(display.HypergraphPathFormatter):
        def label(self, label):
            return "%d %s"%(label[0], tags[label[1]])
        def hyperedge_node_attrs(self, edge):
            return {"color": "pink", "shape": "point"}
        def hypernode_subgraph(self, node):
            return [("cluster_" + str(node.label[0]), None)]
        def subgraph_format(self, subgraph):
            return {"label": (sentence + ["END"])[int(subgraph.split("_")[1])],
                    "rank" : "same"}
        def graph_attrs(self): return {"rankdir":"RL"}
    
        def hypernode_attrs(self, node):
            return {"shape": "",
                    "label": self.label(node.label),
                    "style": "filled",
                    "fillcolor": "#FFFF%d"%(int(((normalized_marg[node.id] - m) / (M-m)) * 100))}
    
    HMMFormat(hypergraph, [path]).to_ipython()



.. image:: hmm_files/hmm_19_0.png



.. code:: python

    for i in range(len(sentence)):
        z = items.item_vector([(i, t) for t in range(len(tags))])
        print normalized_marg  * (item_mat.T * z)

.. parsed-literal::

    [ 1.]
    [ 1.]
    [ 1.]
    [ 1.]
    [ 1.]
    [ 1.]
    [ 1.]
    [ 1.]


.. code:: python

    mat = np.zeros([len(sentence), len(tags)])
    for i, (j, t) in items.iter_items():
        c = item_mat.T * items.item_vector([(j,t)])
        if not c.nonzero()[0]: continue
        mat[j, t] = normalized_marg[c.nonzero()[0][0]]
.. code:: python

    df = pd.DataFrame(mat.T)
    df.columns=sentence
    df.index=tags
    plt.pcolor(df)
    plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
    plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)
    plt.show()


.. image:: hmm_files/hmm_22_0.png

