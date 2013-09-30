
A Constrained HMM Example
-------------------------


.. code:: python

    import pydecode.hyper as ph
    import pydecode.display as display
Step 1: Construct the HMM probabilities.

.. code:: python

    # The emission probabilities.
    emission = {'the' :  {'D': 0.8, 'N': 0.1, 'V': 0.1},
                'dog' :  {'D': 0.1, 'N': 0.8, 'V': 0.1},
                'walked' : {'V': 1},
                'in' :   {'D': 1},
                'park' : {'N': 0.1, 'V': 0.9},
                'END' :  {'END' : 0}}
          
    
    # The transition probabilities.
    transition = {'D' :    {'D' : 0.1, 'N' : 0.8, 'V' : 0.1, 'END' : 0},
                  'N' :    {'D' : 0.1, 'N' : 0.1, 'V' : 0.8, 'END' : 0},
                  'V' :    {'D' : 0.4, 'N' : 0.3, 'V' : 0.3, 'END' : 0},
                  'ROOT' : {'D' : 0.4, 'N' : 0.3, 'V' : 0.3}}
    
    # The sentence to be tagged.
    sentence = 'the dog walked in the park'
Step 2: Construct the hypergraph topology.

.. code:: python

    hypergraph = ph.Hypergraph()                      
    with hypergraph.builder() as b:
        node_start = b.add_node()
        node_list = [(node_start, "ROOT")]
        words = sentence.strip().split(" ") + ["END"]
            
        for word in words:
            next_node_list = []
            for tag in emission[word].iterkeys():
                edges = (([prev_node], (word, tag, prev_tag))
                         for prev_node, prev_tag in node_list)
                node = b.add_node(edges, label = str((word, tag)))
                next_node_list.append((node, tag))
            node_list = next_node_list
Step 3: Construct the weights.

.. code:: python

    def build_weights((word, tag, prev_tag)):
        return transition[prev_tag][tag] + emission[word][tag] 
    weights = ph.Weights(hypergraph, build_weights)
.. code:: python

    # Find the viterbi path.
    path, chart = ph.best_path(hypergraph, weights)
    print weights.dot(path)
    
    # Output the path.
    for edge in path.edges():
        print hypergraph.label(edge)

.. parsed-literal::

    8.6
    ('the', 'D', 'ROOT')
    ('dog', 'N', 'D')
    ('walked', 'V', 'N')
    ('in', 'D', 'V')
    ('the', 'N', 'D')
    ('park', 'V', 'N')
    ('END', 'END', 'V')


.. code:: python

    display.to_ipython(hypergraph, paths=[path])



.. image:: hmm_files/hmm_9_0.png



Step 4: Add the constraints.

.. code:: python

    # The tag of "dog" is the same tag as "park".
    constraints = ph.Constraints(hypergraph)
    for cons_tag in ["D", "V", "N"]:
        def constraint((word, tag, prev_tag)):
            if cons_tag != tag: return 0
            return {"dog" : 1, "park" : -1}.get(word, 0) 
        constraints.add("tag_" + cons_tag, constraint, 0)
This check fails because the tags do not agree.

.. code:: python

    print "check", constraints.check(path)

.. parsed-literal::

    check ['tag_V', 'tag_N']


Solve instead using subgradient.

.. code:: python

    gpath = ph.best_constrained(hypergraph, weights, constraints)
.. code:: python

    # Output the path.
    for edge in gpath.edges():
        print hypergraph.label(edge)
.. code:: python

    print "check", constraints.check(gpath)
    print "score", weights.dot(gpath)
.. code:: python

    display.to_ipython(hypergraph, paths=[path, gpath])



.. image:: hmm_files/hmm_18_0.png


