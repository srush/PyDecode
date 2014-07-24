
.. code:: python

    import sklearn.preprocessing
    import sklearn.metrics
    import pydecode
    import pydecode.model 
    import numpy as np
    from collections import Counter, defaultdict
    import itertools
    from pystruct.learners import StructuredPerceptron
.. code:: python

    np.max(np.max(np.array([[1,2,3],[32,4]]))) + 1



.. parsed-literal::

    33



.. code:: python

    def viterbi(n, K):
        t = max([i for k in K for i in k]) + 1
        items = np.arange(n * t, dtype=np.int64)\
            .reshape([n, t])
        out = np.arange(n * t * t, dtype=np.int64)\
            .reshape([n, t, t])
        c = pydecode.ChartBuilder(items, out)
        c.init(items[0, K[0]])
        for i in range(1, n):
            for t in K[i]:
                c.set(items[i, t],
                      items[i-1, K[i-1]],
                      out=out[i, t, K[i-1]])
        return c.finish()
.. code:: python

    class NGramCoder:
        def __init__(self, n_size):
            self.tag_encoder = sklearn.preprocessing.LabelEncoder()
            self.n_size = n_size
    
        def fit(self, tags):
            self.tag_encoder.fit(tags) 
            return self
    
        def inverse_transform(self, outputs):
            y = [None]
            for output in sorted(outputs, key=lambda a:a[0]):
                y.append(output[1])
                if len(y) == 2:
                    y[0] = output[2]
            return self.tag_encoder.inverse_transform(y)
    
        def transform(self, y):
            tags = self.tag_encoder.transform(y)
            return [(i,) + tuple([tags[i - k] if i - k >= 0 else 0 
                                  for k in range(self.n_size)])
                    for i in range(len(y))]
.. code:: python

    coder = NGramCoder(2).fit(["START", "N", "D", "V", "END"])
    print coder.transform(["N", "D", "N"])
    print coder.transform(["D", "D", "D"])

.. parsed-literal::

    [(0, 2, 0), (1, 0, 2), (2, 2, 0)]
    [(0, 0, 0), (1, 0, 0), (2, 0, 0)]


.. code:: python

    class BigramTagger(pydecode.model.DynamicProgrammingModel):
        def __init__(self, tags):
            coder = NGramCoder(2)
    
            super(BigramTagger, self).__init__(output_coder=coder)
            self.tags = tags + ["START", "END"]
            coder.fit(self.tags)
            
            self._START = coder.tag_encoder.transform(["START"])[0]
            self._END = coder.tag_encoder.transform(["END"])[0]
            self._trans_tags = list(coder.tag_encoder.transform(tags))
    
        def feature_templates(self):
            return [(len(self.tags)),
                    (len(self.tags), len(self.tags))]
    
        def generate_features(self, element, x): 
            i, tag, prev_tag = element
            return [(tag), 
                    (tag, prev_tag)]
          
        def chart(self, x):
            n = len(x)
            K = [[self._START]] + [self._trans_tags] * (n - 2) + [[self._END]]
            return viterbi(len(x), K)
    
        def loss(self, yhat, y):
            print yhat, y
            return sklearn.metrics.hamming_loss(yhat, y)
    
        def max_loss(self, y):
            return len(y)
.. code:: python

    # tag_sequences = [tags.split() for tags in ["D N V", "I D N", "I D N"]]
    data_X = [["START"] + sentence.split()+["END"]  for sentence in 
              ["the dog walked",
               "in the park",
               "in the dog"]]
    data_Y = [["START"] + tags.split() + ["END"] for tags in ["D N V", "I D N", "I D N"]]
    data_X, data_Y



.. parsed-literal::

    ([['START', 'the', 'dog', 'walked', 'END'],
      ['START', 'in', 'the', 'park', 'END'],
      ['START', 'in', 'the', 'dog', 'END']],
     [['START', 'D', 'N', 'V', 'END'],
      ['START', 'I', 'D', 'N', 'END'],
      ['START', 'I', 'D', 'N', 'END']])



.. code:: python

    tagger = BigramTagger(["N", "V", "D", "I"])
    sp = StructuredPerceptron(tagger, verbose=1, max_iter=3)
    
    sp.fit(data_X, data_Y)

.. parsed-literal::

    iteration 0
    ['START', 'D', 'N', 'V', 'END'] ['START' 'N' 'N' 'N' 'END']
    ['START', 'I', 'D', 'N', 'END'] ['START' 'D' 'V' 'V' 'END']
    ['START', 'I', 'D', 'N', 'END'] ['START' 'I' 'D' 'N' 'END']
    avg loss: 0.066667 w: [ 1.  0.  1. -1.  0. -1.  0.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  1.  0.  2.  0.  0. -2. -1.  0.  0.  0.  0.  0.  0.  0.
     -1.  0.  0.  1.  0. -1.]
    effective learning rate: 1.000000
    iteration 1
    ['START', 'D', 'N', 'V', 'END'] ['START' 'I' 'D' 'N' 'END']
    ['START', 'I', 'D', 'N', 'END'] ['START' 'D' 'N' 'V' 'END']
    ['START', 'I', 'D', 'N', 'END'] ['START' 'I' 'D' 'N' 'END']
    avg loss: 0.080000 w: [ 1.  0.  1. -1.  0. -1.  0.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  1.  0.  2.  0.  0. -2. -1.  0.  0.  0.  0.  0.  0.  0.
     -1.  0.  0.  1.  0. -1.]
    effective learning rate: 1.000000
    iteration 2
    ['START', 'D', 'N', 'V', 'END'] ['START' 'I' 'D' 'N' 'END']
    ['START', 'I', 'D', 'N', 'END'] ['START' 'D' 'N' 'V' 'END']
    ['START', 'I', 'D', 'N', 'END'] ['START' 'I' 'D' 'N' 'END']
    avg loss: 0.080000 w: [ 1.  0.  1. -1.  0. -1.  0.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  1.  0.  2.  0.  0. -2. -1.  0.  0.  0.  0.  0.  0.  0.
     -1.  0.  0.  1.  0. -1.]
    effective learning rate: 1.000000




.. parsed-literal::

    StructuredPerceptron(average=False, batch=False, decay_exponent=0,
               decay_t0=10, logger=None, max_iter=3,
               model=BigramTagger, size_joint_feature: 42, n_jobs=1, verbose=1)



.. code:: python

    class TaggingPreprocessor(pydecode.model.SequencePreprocessor):
        WORD = 0
        PREFIX_1 = 1
        PREFIX_2 = 2
        PREFIX_3 = 3
        SUFFIX_1 = 4
        SUFFIX_2 = 5
        SUFFIX_3 = 6
    
        def preprocess_item(self, word):
            return [word, word[:1], word[:2], word[:3], word[-3:], word[-2:], word[-1:]]
.. code:: python

    preprocess = TaggingPreprocessor()
    preprocess.initialize(["the dog brown".split(), "the brown dog".split()])
    preprocess.preprocess("the dog brown".split())



.. parsed-literal::

    [array([2, 1, 0]),
     array([2, 1, 0]),
     array([2, 1, 0]),
     array([2, 1, 0]),
     array([2, 0, 1]),
     array([0, 1, 2]),
     array([0, 1, 2])]



.. code:: python

    class BetterBigramTagger(pydecode.model.DynamicProgrammingModel):
        ENC = TaggingPreprocessor
        def __init__(self, tags, pruner=None):
            coder = NGramCoder(2)
            super(BetterBigramTagger, self).__init__(TaggingPreprocessor(),
                                                     output_coder=coder,
                                                     pruner=pruner)
    
            self.tags = tags + ["START", "END"]
            coder.fit(self.tags)
            
            self._START = coder.tag_encoder.transform(["START"])[0]
            self._END = coder.tag_encoder.transform(["END"])[0]
            self._trans_tags = list(coder.tag_encoder.transform(tags))
    
        
    
        def feature_templates(self):
            def size(t):
                return self._preprocessor.size(t)
            return [(len(self.tags), size(self.ENC.WORD)),
                    (len(self.tags), size(self.ENC.SUFFIX_1)),
                    (len(self.tags), size(self.ENC.SUFFIX_2)),
                    (len(self.tags), size(self.ENC.SUFFIX_3)),
                    (len(self.tags), size(self.ENC.PREFIX_1)),
                    (len(self.tags), size(self.ENC.PREFIX_2)),
                    (len(self.tags), size(self.ENC.PREFIX_3)),
                    (len(self.tags), len(self.tags))
                    ]
    
        def generate_features(self, element, x): 
            i, tag, prev_tag = element
            return [(tag, x[self.ENC.WORD][i]),
                    (tag, x[self.ENC.SUFFIX_1][i]),
                    (tag, x[self.ENC.SUFFIX_2][i]),
                    (tag, x[self.ENC.SUFFIX_3][i]),
                    (tag, x[self.ENC.PREFIX_1][i]),
                    (tag, x[self.ENC.PREFIX_2][i]),
                    (tag, x[self.ENC.PREFIX_3][i]),
                    (tag, prev_tag),
                    ]
    
        def chart(self, x):
            n = len(x)
            K = [[self._START]] + [self._trans_tags] * (n - 2) + [[self._END]]
            return viterbi(len(x), K)
    
        def loss(self, yhat, y):
            print yhat, y
            return sklearn.metrics.hamming_loss(yhat, y)
    
        def max_loss(self, y):
            return len(y)

.. code:: python

    tagger = BetterBigramTagger(["N", "V", "D", "I"])
    sp = StructuredPerceptron(tagger, verbose=1, max_iter=5)
    sp.fit(data_X, data_Y)

.. parsed-literal::

    iteration 0
    ['START', 'D', 'N', 'V', 'END'] ['START' 'N' 'N' 'N' 'END']
    ['START', 'I', 'D', 'N', 'END'] ['START' 'D' 'D' 'V' 'END']
    ['START', 'I', 'D', 'N', 'END'] ['START' 'I' 'D' 'N' 'END']
    avg loss: 0.053333 w: [ 0.  0.  0. -1.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.
      0.  0.  0.  0.  0.  0.  0.  1. -1. -1.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0. -1.  0.  1.  0.  0.  0.  0. -1.  0.  1.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  0.  1.  0.  0. -1. -1.  0.  0.
      0.  0.  0.  0.  0.  0.  0. -1.  0.  0.  1.  0.  0.  0.  0.  1. -1.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  0. -1.
     -1.  0.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  0.  0.  0. -1.
      0.  0.  0.  1.  0.  0. -1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  1.  0.  0. -1. -1.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  1.  0.  0. -1.  0.  0.  0.  0. -1.  0.  1.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.  0.  0.  1. -1. -1.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0. -1.  0.  1.  0.  0.  0. -1.  0.  1.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.  0.
      0.  1. -1. -1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0. -1.  0.  1.
      0.  0.  0. -1.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.
      0.  0.  0.  0.  0.  0.  0.  1. -1. -1.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0. -1.  0.  1. -1.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  1.  0.  2.  0.  0. -2. -1.  0.  0.  0.  0.  0.  0.  0.
     -1.  0.  0.  1.  0.  0.]
    effective learning rate: 1.000000
    iteration 1
    ['START', 'D', 'N', 'V', 'END'] ['START' 'D' 'N' 'V' 'END']
    ['START', 'I', 'D', 'N', 'END'] ['START' 'I' 'D' 'N' 'END']
    ['START', 'I', 'D', 'N', 'END'] ['START' 'I' 'D' 'N' 'END']
    avg loss: 0.000000 w: [ 0.  0.  0. -1.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.
      0.  0.  0.  0.  0.  0.  0.  1. -1. -1.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0. -1.  0.  1.  0.  0.  0.  0. -1.  0.  1.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  0.  1.  0.  0. -1. -1.  0.  0.
      0.  0.  0.  0.  0.  0.  0. -1.  0.  0.  1.  0.  0.  0.  0.  1. -1.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  0. -1.
     -1.  0.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  0.  0.  0. -1.
      0.  0.  0.  1.  0.  0. -1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  1.  0.  0. -1. -1.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  1.  0.  0. -1.  0.  0.  0.  0. -1.  0.  1.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.  0.  0.  1. -1. -1.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0. -1.  0.  1.  0.  0.  0. -1.  0.  1.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.  0.
      0.  1. -1. -1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0. -1.  0.  1.
      0.  0.  0. -1.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.
      0.  0.  0.  0.  0.  0.  0.  1. -1. -1.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0. -1.  0.  1. -1.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  1.  0.  2.  0.  0. -2. -1.  0.  0.  0.  0.  0.  0.  0.
     -1.  0.  0.  1.  0.  0.]
    effective learning rate: 1.000000
    Loss zero. Stopping.




.. parsed-literal::

    StructuredPerceptron(average=False, batch=False, decay_exponent=0,
               decay_t0=10, logger=None, max_iter=5,
               model=BetterBigramTagger, size_joint_feature: 330, n_jobs=1,
               verbose=1)



.. code:: python

    def sentences(file):
        sentence = []
        for l in open(file):
            t = l.strip().split()
            if len(t) == 2:
                sentence.append(t)
            else:
                yield sentence
                sentence = []
        yield sentence
.. code:: python

    # sents = [zip(*sentence) for sentence in sentences("data/tag_train_small.dat")]  
    # X, Y = zip(*sents)
    # tags = set()
    # for t in Y:
    #     tags.update(t)

.. code:: python

    # tagger = BetterBigramTagger(list(tags))
    # sp = StructuredPerceptron(tagger, verbose=1, max_iter=5)
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    #     sp.fit(X, Y)

.. code:: python

    class DictionaryPruner:
        def __init__(self, limit=1):
            self._limit = limit
        
        def initialize(self, X, Y, output_coder):
            self._word_tag_counts = defaultdict(Counter)
            self._word_counts = Counter()
            for x, y in itertools.izip(X, Y):
                elements = output_coder.transform(y)
                for element in elements:
                    self._word_tag_counts[x[element[0]]][element[1]] += 1
                    self._word_counts[x[element[0]]] += 1
            self._all_tags = range(output_coder.n_size)
    
        def table(self, x):
            table = [[0]]
            for word in x:
                if self._word_counts[word] < self._limit:
                    table.append(self._all_tags)
                else:
                    table.append(self._word_tag_counts[word].keys())
            return table
.. code:: python

    pruner = DictionaryPruner()
    coder = NGramCoder(2).fit(["N", "V"])
    pruner.initialize([["hi", "you"]], [["N", "V"]], coder)
    pruner.table(["hi", "you", "hi"])



.. parsed-literal::

    [[0], [0], [1], [0]]



.. code:: python

    class PrunedBigramTagger(BetterBigramTagger):
        def chart(self, x):
            table = self._pruner.table(x)
            return viterbi(len(x), table)
.. code:: python

    # def pruned_bigram_tagger(n, tag_sets):
    #     max_size = max([max(tag_set) for tag_set in tag_sets])
    #     c = pydecode.ChartBuilder(item_set=pydecode.IndexSet((n+2, max_size + 1)), 
    #                         output_set=output_set(n, max_size + 1))
    #     for tag in tag_sets[0]:
    #         c[0, tag] = c.init()
    
    #     for i in range(1, n+1):
    #         for tag in tag_sets[i]:
    #             c[i, tag] = \
    #                 [c.merge((i-1, prev), values=[(i-1, tag, prev)])
    #                  for prev in tag_sets[i-1]]
    
    #     c[n+1, 0] = [c.merge((n, prev), values=[]) 
    #                  for prev in tag_sets[n]]
    #     return c
.. code:: python

    # tagger = PrunedBigramTagger(list(tags), pruner = DictionaryPruner(1000))
    # sp = StructuredPerceptron(tagger, verbose=1, max_iter=3)
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    #     sp.fit(X[:50], Y[:50])
.. code:: python

    # output = sp.predict(X[:100])
    # output
    # import sklearn.metrics
    # sklearn.metrics.hamming_loss([o.tolist() for o in output][:100], Y[:100])