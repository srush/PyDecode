
Sequence CRF
============


.. code:: python

    from sklearn.feature_extraction import DictVectorizer
    from collections import namedtuple
    import pydecode.model as model
    import pydecode.chart as chart
.. code:: python

    # The emission probabilities.
    emission = {'ROOT' : {'ROOT' : 1.0},
                'the' :  {'D': 0.8, 'N': 0.1, 'V': 0.1},
                'dog' :  {'D': 0.1, 'N': 0.8, 'V': 0.1},
                'walked':{'V': 1},
                'in' :   {'D': 1},
                'park' : {'N': 0.1, 'V': 0.9},
                'END' :  {'END' : 1.0}}
    
    # The transition probabilities.
    transition = {'D' :    {'D' : 0.1, 'N' : 0.8, 'V' : 0.1, 'END' : 0},
                  'N' :    {'D' : 0.1, 'N' : 0.1, 'V' : 0.8, 'END' : 0},
                  'V' :    {'D' : 0.4, 'N' : 0.3, 'V' : 0.3, 'END' : 0},
                  'ROOT' : {'D' : 0.4, 'N' : 0.3, 'V' : 0.3}}
.. code:: python

    class Bigram(namedtuple("Bigram", ["position", "tag", "prevtag"])):
        def __str__(self): return "%s -> %s"%(self.prevtag, self.tag)
        
        @staticmethod
        def from_tagging(tagging):
            return [Bigram(i, tag=tag, prevtag=tagging[i-1])
                    for i, tag in enumerate(tagging)]
          
    class Tagged(namedtuple("Tagged", ["position", "word", "tag"])):
        def __str__(self): return "%s %s"%(self.word, self.tag)
.. code:: python

    def sequence_dynamic_program(sentence, c):
        words = ["ROOT"] + sentence + ["END"]
        c.init(Tagged(0, "ROOT", "ROOT"))
        for i, word in enumerate(words[1:], 1):
            prev_tags = emission[words[i-1]].keys()
            for tag in emission[word].iterkeys():
                c[Tagged(i, word, tag)] = \
                    c.sum([c[key] * c.sr(Bigram(i - 2, tag, prev))
                           for prev in prev_tags 
                           for key in [Tagged(i - 1, words[i - 1], prev)] 
                           if key in c])
        return c
.. code:: python

    c = chart.ChartBuilder(lambda a:a, chart.HypergraphSemiRing, 
                           build_hypergraph = True)
    hypergraph = sequence_dynamic_program(["the", "dog"], c).finish()
    for edge in hypergraph.edges:
        print hypergraph.label(edge)

.. parsed-literal::

    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    ROOT -> V
    ROOT -> D
    ROOT -> N
    V -> V
    D -> V
    N -> V
    V -> D
    D -> D
    N -> D
    V -> N
    D -> N
    N -> N
    V -> END
    D -> END
    N -> END


.. code:: python

    class TaggingCRFModel(model.DynamicProgrammingModel):
        def dynamic_program(self, sentence, c):
            return sequence_dynamic_program(sentence, c) 
    
        def factored_psi(self, sentence, bigram):
            print bigram, sentence
            return {#"word-1:%s"%sentence[bigram.position - 1] if bigram.position != 0 else "", 
                    "word:%s" % sentence[bigram.position], 
                    "tag-1:%s" % bigram.prevtag, 
                    "tag:%s" % bigram.tag}
.. code:: python

    data_X = map(lambda a: a.split(),
                 ["the dog walked END",
                  "in the park END",
                  "in the dog END"])
    data_Y = map(lambda a: Bigram.from_tagging(a.split()),
                 ["D N V", "I D N", "I D N"])
    
    hm = TaggingCRFModel()
.. code:: python

    from pystruct.learners import StructuredPerceptron  
    sp = StructuredPerceptron(hm)
    sp.fit(data_X, data_Y)

.. parsed-literal::

    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    ROOT -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

    
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

    
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V


.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

    
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']


.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make 

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

    N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D 

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

    ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N


.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END


.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V 

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

    ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END


.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

    
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make 

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

    V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make 

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

    N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    make ROOT -> V
    make ROOT -> D
    make ROOT -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> END
    make END -> END
    ROOT -> V ['the', 'dog', 'walked', 'END']
    ROOT -> D ['the', 'dog', 'walked', 'END']
    ROOT -> N

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))


.. parsed-literal::

     ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> D ['the', 'dog', 'walked', 'END']
    N -> D ['the', 'dog', 'walked', 'END']
    V -> N ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    V -> V ['the', 'dog', 'walked', 'END']
    D -> V ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    V -> D ['the', 'dog', 'walked', 'END']
    D -> N ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    N -> N ['the', 'dog', 'walked', 'END']
    ROOT -> N ['the', 'dog', 'walked', 'END']
    V -> END ['the', 'dog', 'walked', 'END']
    END -> END ['the', 'dog', 'walked', 'END']
    N -> V ['the', 'dog', 'walked', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    V -> V ['in', 'the', 'park', 'END']
    D -> V ['in', 'the', 'park', 'END']
    N -> V ['in', 'the', 'park', 'END']
    V -> N ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    N -> N ['in', 'the', 'park', 'END']
    V -> END ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    N -> I ['in', 'the', 'park', 'END']
    I -> D ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    ROOT -> D ['in', 'the', 'park', 'END']
    D -> D ['in', 'the', 'park', 'END']
    N -> END ['in', 'the', 'park', 'END']
    D -> N ['in', 'the', 'park', 'END']
    END -> END ['in', 'the', 'park', 'END']
    make ROOT -> D
    make D -> V
    make D -> D
    make D -> N
    make V -> V
    make D -> V
    make N -> V
    make V -> D
    make D -> D
    make N -> D
    make V -> N
    make D -> N
    make N -> N
    make V -> END
    make D -> END
    make N -> END
    make END -> END
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    V -> V ['in', 'the', 'dog', 'END']
    D -> V ['in', 'the', 'dog', 'END']
    N -> V ['in', 'the', 'dog', 'END']
    V -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    N -> D ['in', 'the', 'dog', 'END']
    V -> N ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    N -> N ['in', 'the', 'dog', 'END']
    V -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']
    N -> END ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    N -> I ['in', 'the', 'dog', 'END']
    I -> D ['in', 'the', 'dog', 'END']
    D -> N ['in', 'the', 'dog', 'END']
    ROOT -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    D -> D ['in', 'the', 'dog', 'END']
    END -> END ['in', 'the', 'dog', 'END']
    D -> END ['in', 'the', 'dog', 'END']


.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))
    /usr/local/lib/python2.7/dist-packages/pystruct/learners/structured_perceptron.py:149: DeprecationWarning: Implicitly casting between incompatible kinds. In a future numpy release, this will raise an error. Use casting="unsafe" if this is intentional.
      self.model.psi(x, y_hat))




.. parsed-literal::

    StructuredPerceptron(average=False, batch=False, decay_exponent=0,
               decay_t0=10, logger=None, max_iter=100,
               model=TaggingCRFModel, size_psi: 13, n_jobs=1, verbose=0)


