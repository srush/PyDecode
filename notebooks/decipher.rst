
A Decipherment Example
======================


This is a note on running decipherment.

.. code:: python

    from nltk.util import ngrams
    from nltk.model.ngram import NgramModel
    from nltk.probability import LidstoneProbDist
    import random, math
    
    class Problem:
        def __init__(self, corpus):
            self.t = corpus
            t = list(self.t)
            est = lambda fdist, bins: LidstoneProbDist(fdist, 0.0001)
            self.lm = NgramModel(2, t, estimator = est)
    
            self.letters = set(t) #[chr(ord('a') + i) for i in range(26)]
            self.letters.remove(" ")
            shuffled = list(self.letters)
            random.shuffle(shuffled)
            self.substitution_table = dict(zip(self.letters, shuffled))
            self.substitution_table[" "] = " "
    
        def make_cipher(self, plaintext):
            self.ciphertext = "".join([self.substitution_table[l] for l in plaintext])
            self.plaintext = plaintext
    simple_problem = Problem("ababacabac ")
    simple_problem.make_cipher("abac")
.. code:: python

    import pydecode.hyper as hyper
    import pydecode.display as display
    import pydecode.constraints as cons
    from collections import namedtuple        
.. code:: python

    class Conversion(namedtuple("Conversion", ["i", "cipherletter", "prevletter", "letter"])):
        __slots__ = ()
        def __str__(self):
            return "%s %s %s"%(self.cipherletter, self.prevletter, self.letter)
    class Node(namedtuple("Node", ["i", "cipherletter", "letter"])):
        __slots__ = ()
        def __str__(self):
            return "%s %s %s"%(self.i, self.cipherletter, self.letter)
.. code:: python

    def build_cipher_graph(problem):
        ciphertext = problem.ciphertext
        letters = problem.letters
        hypergraph = hyper.Hypergraph()
        with hypergraph.builder() as b:
            prev_nodes = [(" ", b.add_node([], label=Node(-1, "", "")))]
            for i, c in enumerate(ciphertext):
                nodes = []
                possibilities = letters
                if c == " ": possibilities = [" "]
                for letter in possibilities:
                    edges = [([prev_node], Conversion(i, c, old_letter, letter))
                             for (old_letter, prev_node) in prev_nodes]
                    
                    node = b.add_node(edges, label = Node(i, c, letter))
                    nodes.append((letter, node))
                prev_nodes = nodes
            letter = " "
            final_edges = [([prev_node], Conversion(i, c, old_letter, letter))
                           for (old_letter, prev_node) in prev_nodes]
            b.add_node(final_edges, label=Node(len(ciphertext), "", ""))
        return hypergraph
.. code:: python

    hyper1 = build_cipher_graph(simple_problem)
.. code:: python

    class CipherFormat(display.HypergraphPathFormatter):
        def hypernode_attrs(self, node):
            label = self.hypergraph.node_label(node)
            return {"label": "%s -> %s"%(label.cipherletter, label.letter)}
        def hyperedge_node_attrs(self, edge):
            return {"color": "pink", "shape": "point"}
        def hypernode_subgraph(self, node):
            label = self.hypergraph.node_label(node)
            return [("cluster_" + str(label.i), label.i)]
        # def subgraph_format(self, subgraph):
        #     return {"label": (sentence.split() + ["END"])[int(subgraph.split("_")[1])]}
    
    CipherFormat(hyper1, []).to_ipython()



.. image:: decipher_files/decipher_7_0.png



.. code:: python

    
Constraint is that the sum of edges with the conversion is equal to the
0.

l^2 constraints

.. code:: python

    def build_constraints(hypergraph, problem):
        ciphertext = problem.ciphertext
        letters = problem.letters
        def transform(from_l, to_l): return "letter_%s_from_letter_%s"%(to_l, from_l)
        constraints = cons.Constraints(hypergraph, [(transform(l, l2), 0)
                           for l  in letters 
                           for l2 in letters])
    
        first_position = {}
        count = {}
        for i, l in enumerate(ciphertext):
            if l not in first_position:
                first_position[l] = i
            count.setdefault(l, 0)
            count[l] += 1
        def build(conv):
            l = conv.cipherletter
            if l == " ": return []
            if conv.letter == " ": return []
            if first_position[l] == conv.i:
                return [(transform(conv.cipherletter, conv.letter), count[l] - 1)]
            else:
                return [(transform(conv.cipherletter, conv.letter), -1)]
        constraints.build( 
                          build)
        return constraints
    constraints = build_constraints(hyper1, simple_problem)

.. code:: python

    def build_potentials(edge):
        return random.random()
    potentials = hyper.Potentials(hyper1).build(build_potentials)
.. code:: python

    for edge in hyper1.edges:
        print potentials[edge]

.. parsed-literal::

    0.614117264748
    0.759764134884
    0.733525454998
    0.0815630927682
    0.916156351566
    0.0582501739264
    0.389200419188
    0.708899497986
    0.117828272283
    0.703362822533
    0.955223083496
    0.951672554016
    0.889782249928
    0.850794136524
    0.206611990929
    0.666242241859
    0.836310505867
    0.631427586079
    0.420347988605
    0.717808246613
    0.0318541526794
    0.10944814235
    0.398276388645
    0.686978042126
    0.0620245561004
    0.156915932894
    0.227964177728
    0.761591732502
    0.0153166977689
    0.402361214161
    0.468028366566
    0.351031720638
    0.130741521716


.. code:: python

    path = hyper.best_path(hyper1, potentials)
    potentials.dot(path)



.. parsed-literal::

    3.458035945892334



.. code:: python

    import pydecode.optimization as opt
    cpath = opt.best_constrained_path(hyper1, potentials, constraints)
.. code:: python

    CipherFormat(hyper1, [cpath]).to_ipython()



.. image:: decipher_files/decipher_15_0.png



.. code:: python

    print potentials.dot(cpath)
    constraints.check(cpath)

.. parsed-literal::

    0.468028366566
    Constraints {}




.. parsed-literal::

    []



Real Problem

.. code:: python

    complicated_problem = Problem("this is the president calling blah blah abadadf adfadf")
    complicated_problem.make_cipher("this is the president calling")
.. code:: python

    hyper2 = build_cipher_graph(complicated_problem)
.. code:: python

    def build_ngram_potentials(edge):
        return math.log(complicated_problem.lm.prob(edge.letter, edge.prevletter))
    potentials2 = hyper.Potentials(hyper2).build(build_ngram_potentials)

.. code:: python

    print len(hyper2.edges)

.. parsed-literal::

    4650


.. code:: python

    path2 = hyper.best_path(hyper2, potentials2)
    
    for edge in path2.edges:
        print edge.id
        print potentials2[edge]
    potentials2.dot(path2)

.. parsed-literal::

    11
    -2.07941651344
    221
    0.0
    298
    0.0
    648
    -1.0986123085
    702
    -0.405481785536
    709
    -1.45088791847
    814
    -0.510852277279
    951
    -0.693147182465
    971
    -2.07941651344
    1181
    0.0
    1258
    0.0
    1428
    -1.0986123085
    1451
    -2.07941651344
    1661
    0.0
    1738
    0.0
    1908
    -0.693234682083
    2190
    -0.693172156811
    2449
    -0.510852277279
    2586
    -0.693147182465
    2865
    -0.693172156811
    3124
    -0.510852277279
    3261
    -0.693147182465
    3281
    -2.07941651344
    3491
    0.0
    3568
    0.0
    3888
    -1.0986123085
    3970
    -0.693234682083
    4245
    -0.693172156811
    4504
    -0.510852277279
    4641
    -0.693147182465




.. parsed-literal::

    0.0



.. code:: python

    # new_hyper, new_potentials = hyper.prune_hypergraph(hyper2, potentials2, 0.2)
    # constraints2 = build_constraints(new_hyper, complicated_problem)
.. code:: python

    # print hyper2.edges_size
    # new_hyper.edges_size
.. code:: python

    # display.report(duals)
.. code:: python

    # path2, duals = hyper.best_constrained(new_hyper, new_potentials, constraints2)
Potentials are the bigram language model scores.

.. code:: python

    # path2 = hyper.best_path(hyper2, potentials2)
    # print potentials2.dot(path2)
    # for edge in path2.edges:
    #     print hyper2.label(edge).letter, 