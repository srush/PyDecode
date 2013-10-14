
Decipherment Note
=================


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
            return ["cluster_" + str(label.i)]
        # def subgraph_format(self, subgraph):
        #     return {"label": (sentence.split() + ["END"])[int(subgraph.split("_")[1])]}
    
    display.to_ipython(hyper1, CipherFormat(hyper1, []))



.. image:: decipher_files/decipher_7_0.png



.. code:: python

    
Constraint is that the sum of edges with the conversion is equal to the
0.

l^2 constraints

.. code:: python

    def build_constraints(hypergraph, problem):
        ciphertext = problem.ciphertext
        letters = problem.letters
        constraints = hyper.Constraints(hypergraph)
        def transform(from_l, to_l): return "letter_%s_from_letter_%s"%(to_l, from_l)
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
        constraints.build([(transform(l, l2), 0)
                           for l  in letters 
                           for l2 in letters], 
                          build)
        return constraints
    constraints = build_constraints(hyper1, simple_problem)

.. code:: python

    def build_weights(edge):
        return random.random()
    weights = hyper.Weights(hyper1).build(build_weights)
.. code:: python

    for edge in hyper1.edges:
        print weights[edge]

.. parsed-literal::

    0.70303262896
    0.676212295905
    0.43750593877
    0.00938915686059
    0.564162079225
    0.722017722152
    0.247006890533
    0.539191293399
    0.216991331549
    0.730933205951
    0.469229775434
    0.371490981225
    0.982409886818
    0.221561306386
    0.778218323839
    0.0656559254133
    0.468223740379
    0.520127234634
    0.921828628809
    0.282919306759
    0.926934958024
    0.22385320216
    0.184922369718
    0.890985305051
    0.996326784576
    0.16965504918
    0.564655548258
    0.221743806149
    0.304515879722
    0.292258405864
    0.409225055659
    0.618844153235
    0.249232775945


.. code:: python

    path, _ = hyper.best_path(hyper1, weights)
    weights.dot(path)



.. parsed-literal::

    3.837955199760571



.. code:: python

    cpath, duals = hyper.best_constrained(hyper1, weights, constraints)
.. code:: python

    display.to_ipython(hyper1, CipherFormat(hyper1, [cpath]))



.. image:: decipher_files/decipher_15_0.png



.. code:: python

    for d in duals:
        print d.dual

.. parsed-literal::

    3.83795519976
    4.56111115365
    4.47710448555
    4.46159227196
    3.82735509656


.. code:: python

    display.report(duals)


.. image:: decipher_files/decipher_17_0.png


.. code:: python

    print weights.dot(cpath)
    constraints.check(cpath)

.. parsed-literal::

    3.82735509656




.. parsed-literal::

    []



Real Problem
============


.. code:: python

    complicated_problem = Problem("this is the president calling blah blah abadadf adfadf")
    complicated_problem.make_cipher("this is the president calling")
.. code:: python

    hyper2 = build_cipher_graph(complicated_problem)
.. code:: python

    def build_ngram_weights(edge):
        return math.log(complicated_problem.lm.prob(edge.letter, edge.prevletter))
    weights2 = hyper.Weights(hyper2).build(build_ngram_weights)

.. code:: python

    print len(hyper2.edges)

.. parsed-literal::

    4650


.. code:: python

    path2, _ = hyper.best_path(hyper2, weights2)
    
    for edge in path2.edges:
        print edge.id
        print weights2[edge]
    weights2.dot(path)

.. parsed-literal::

    11
    -2.07941654387
    221
    0.0
    298
    0.0
    648
    -1.09861228867
    702
    -0.405481773803
    709
    -1.45088787965
    814
    -0.510852289188
    951
    -0.69314718056
    971
    -2.07941654387
    1181
    0.0
    1258
    0.0
    1428
    -1.09861228867
    1451
    -2.07941654387
    1661
    0.0
    1738
    0.0
    1908
    -0.693234675638
    2190
    -0.693172179622
    2449
    -0.510852289188
    2586
    -0.69314718056
    2865
    -0.693172179622
    3124
    -0.510852289188
    3261
    -0.69314718056
    3281
    -2.07941654387
    3491
    0.0
    3568
    0.0
    3888
    -1.09861228867
    3970
    -0.693234675638
    4245
    -0.693172179622
    4504
    -0.510852289188
    4641
    -0.69314718056




.. parsed-literal::

    -8.957765495182873



.. code:: python

    new_hyper, new_weights = hyper.prune_hypergraph(hyper2, weights2, 0.2)
    constraints2 = build_constraints(new_hyper, complicated_problem)

.. parsed-literal::

    0 0 46 0 381
    1 1 46 1 381
    2 2 46 2 381
    3 3 46 3 381
    4 4 46 4 381
    5 5 46 5 381
    6 16 46 16 381
    7 31 46 31 381
    8 46 46 46 381
    9 50 46 50 381
    10 52 46 52 381
    11 53 46 53 381
    12 54 46 54 381
    13 55 46 55 381
    14 58 46 58 381
    15 61 46 61 381
    16 62 46 62 381
    17 63 46 63 381
    18 64 46 64 381
    19 65 46 65 381
    20 66 46 66 381
    21 77 46 77 381
    22 92 46 92 381
    23 93 46 93 381
    24 94 46 94 381
    25 95 46 95 381
    26 96 46 96 381
    27 97 46 97 381
    28 108 46 108 381
    29 123 46 123 381
    30 138 46 138 381
    31 139 46 139 381
    32 140 46 140 381
    33 141 46 141 381
    34 142 46 142 381
    35 143 46 143 381
    36 154 46 154 381
    37 173 46 173 381
    38 177 46 177 381
    39 184 46 184 381
    40 199 46 199 381
    41 218 46 218 381
    42 222 46 222 381
    43 229 46 229 381
    44 244 46 244 381
    45 267 46 267 381
    0 0
    1 1
    2 2
    3 3
    4 4
    5 15
    6 16
    7 17
    8 18
    9 19
    10 240
    11 465
    12 525
    13 555
    14 570
    15 585
    16 600
    17 645
    18 690
    19 694
    20 696
    21 697
    22 698
    23 699
    24 702
    25 705
    26 706
    27 707
    28 708
    29 709
    30 720
    31 721
    32 722
    33 723
    34 724
    35 945
    36 960
    37 961
    38 962
    39 963
    40 964
    41 975
    42 976
    43 977
    44 978
    45 979
    46 1200
    47 1425
    48 1440
    49 1441
    50 1442
    51 1443
    52 1444
    53 1455
    54 1456
    55 1457
    56 1458
    57 1459
    58 1740
    59 1800
    60 1909
    61 1913
    62 2130
    63 2415
    64 2475
    65 2584
    66 2588
    67 2805
    68 3150


.. code:: python

    print hyper2.edges_size()
    new_hyper.edges_size()

.. parsed-literal::

    4650




.. parsed-literal::

    69



.. code:: python

    #display.to_ipython(new_hyper, CipherFormat(new_hyper, []))
.. code:: python

    display.report(duals)


.. image:: decipher_files/decipher_28_0.png


.. code:: python

    for d in duals[:10]:
        for const in d.constraints:
            print const.label,
        print 

.. parsed-literal::

    letter_a_from_letter_a letter_a_from_letter_c letter_d_from_letter_c letter_a_from_letter_b letter_c_from_letter_b letter_c_from_letter_g letter_h_from_letter_g letter_a_from_letter_l letter_d_from_letter_l
    letter_a_from_letter_a letter_a_from_letter_c letter_h_from_letter_c letter_a_from_letter_b letter_c_from_letter_g letter_b_from_letter_g letter_h_from_letter_g
    letter_a_from_letter_a letter_a_from_letter_c letter_h_from_letter_c letter_a_from_letter_b letter_c_from_letter_g letter_b_from_letter_g letter_h_from_letter_g
    letter_a_from_letter_a letter_a_from_letter_c letter_d_from_letter_c letter_a_from_letter_b letter_c_from_letter_g letter_e_from_letter_g letter_h_from_letter_g
    letter_a_from_letter_a letter_a_from_letter_c letter_d_from_letter_c letter_a_from_letter_b letter_a_from_letter_g letter_c_from_letter_g letter_h_from_letter_g
    letter_a_from_letter_a letter_a_from_letter_c letter_h_from_letter_c letter_a_from_letter_b letter_c_from_letter_g letter_e_from_letter_g letter_h_from_letter_g
    letter_a_from_letter_a letter_a_from_letter_c letter_h_from_letter_c letter_a_from_letter_b letter_c_from_letter_g letter_d_from_letter_g letter_h_from_letter_g
    letter_a_from_letter_a letter_a_from_letter_c letter_d_from_letter_c letter_a_from_letter_b letter_c_from_letter_g letter_h_from_letter_g
    letter_a_from_letter_a letter_a_from_letter_c letter_d_from_letter_c letter_a_from_letter_b letter_c_from_letter_g letter_e_from_letter_g letter_h_from_letter_g
    letter_a_from_letter_a letter_a_from_letter_c letter_h_from_letter_c letter_a_from_letter_b letter_c_from_letter_g letter_e_from_letter_g letter_h_from_letter_g


.. code:: python

    path2, duals = hyper.best_constrained(new_hyper, new_weights, constraints2)
.. code:: python

    print len(duals)

.. parsed-literal::

    200


Weights are the bigram language model scores.

.. code:: python

    path2, _ = hyper.best_path(hyper2, weights2)
    print weights2.dot(path2)
    for edge in path2.edges:
        print hyper2.label(edge).letter, 

.. parsed-literal::

    -21.7518564641
    p r e s   d f   p r e   p r e a d f a d f   p r e n a d f  


.. code:: python

    for edge in path2.edges:
        print new_hyper.label(edge).letter, 


.. parsed-literal::

    c a a a   a a   c a a   c a h a a d a a h


.. code:: python

    new_hyper, new_weights = hyper.prune_hypergraph(hyper2, weights2, 0.9)
    new_constraints = build_constraints(new_hyper, ciphertext2)

::


    ---------------------------------------------------------------------------
    NameError                                 Traceback (most recent call last)

    <ipython-input-28-9ec05a252bb9> in <module>()
          1 new_hyper, new_weights = hyper.prune_hypergraph(hyper2, weights2, 0.9)
    ----> 2 new_constraints = build_constraints(new_hyper, ciphertext2)
    

    NameError: name 'ciphertext2' is not defined


.. parsed-literal::

    0 0 2 0 381
    1 12 2 12 381
    0 11


.. code:: python

    path2, duals = hyper.best_constrained(new_hyper, new_weights, new_constraints)
    # print weights2.dot(path2)
    # for edge in path2.edges:
    #     print hyper2.label(edge).letter, 
.. code:: python

    display.report(duals)