
Tutorial 5: Training a CRF
==========================


.. code:: python

    from sklearn.feature_extraction import DictVectorizer
    from collections import namedtuple
    import pydecode.model as model
    import pydecode.chart as chart
    from collections import Counter, defaultdict
    from itertools import izip
.. code:: python

    class Dictionary:
        def __init__(self, counts, word_counts, tag_set):
            self.counts = counts 
            self.word_counts = word_counts
            self.tag_set = tag_set
            self.tag_num = {tag:i for i, tag in enumerate(tag_set)}
            self.word_num = {word:i for i, word in enumerate(word_counts.iterkeys())}
    
        def emission(self, word):
            if word == "ROOT": return ["<t>"]
            if word == "END": return ["</t>"]
            if self.word_counts[word] > 5:
                return self.counts[word].keys()
            return self.tag_set
    
        def tag_id(self, tag):
            return self.tag_num.get(tag, 0)
    
        def word_id(self, word):
            return self.word_num.get(word, 0)
    
        @staticmethod
        def make(sentences, taggings):
            tag_set = set()
            word_counts = Counter()
            counts = defaultdict(Counter)
            for sentence, tags in izip(sentences, taggings):
                #print sentence, tags
                for word, tag in izip(sentence, tags):
                    counts[word][tag.tag] += 1
                    word_counts[word] += 1
                    tag_set.add(tag.tag)
            print tag_set
            return Dictionary(counts, word_counts, tag_set)
    
        
.. code:: python

    class Bigram(namedtuple("Bigram", ["position", "prevtag", "tag"])):
        def __str__(self): return "%s -> %s"%(self.prevtag, self.tag)
        
        @staticmethod
        def from_tagging(tagging):
            return [Bigram(i, tag=tag, prevtag=tagging[i-1] if i > 0 else "<t>")
                    for i, tag in enumerate(tagging)] + [Bigram(len(tagging), tag="</t>", prevtag=tagging[-1])] 
          
    class Tagged(namedtuple("Tagged", ["position",  "tag"])):
        def __str__(self): return "%s"%(self.tag,)
.. code:: python

    class TaggingCRFModel(model.DynamicProgrammingModel):
        def initialize(self, sentences, tags):
            self.dictionary = Dictionary.make(sentences, tags)
            super(TaggingCRFModel, self).initialize(sentences, tags)
    
    
        def dynamic_program(self, sentence, c):
            words = ["ROOT"] + sentence + ["END"]
            c.init(Tagged(0, "<t>"))
            for i, word in enumerate(words[1:], 1):
                prev_tags = self.dictionary.emission(words[i-1])
                for tag in self.dictionary.emission(word):
                    c[Tagged(i, tag)] = \
                        c.sum([c[key] * c.sr(Bigram(i - 1, prev, tag))
                               for prev in prev_tags 
                               for key in [Tagged(i - 1, prev)] 
                               if key in c])
            return c
    
        def initialize_features(self, sentence):
            return [self.dictionary.word_id(word) for word in sentence]
    
        def factored_psi(self, sentence, bigram, data):
            # return {(1, self.dictionary.tag_id(bigram.tag), data[bigram.position]) : 1,
            #         (2, data[bigram.position]) : 1,
            #         (3, self.dictionary.tag_id(bigram.tag)) : 1,
            #         (4, self.dictionary.tag_id(bigram.prevtag)): 1,
            #         (5, self.dictionary.tag_id(bigram.prevtag), self.dictionary.tag_id(bigram.tag)) : 1
            #         }
            word = sentence[bigram.position] if bigram.position < len(sentence) else "END"
            return {#"word-1:%s"%sentence[bigram.position - 1] if bigram.position != 0 else "", 
                    "word:tag:%s:%s" % (bigram.tag, word) : 1, 
                    "word:%s" %  word : 1, 
                    "tag-1:%s" % bigram.prevtag : 1, 
                    "tag:%s" % bigram.tag : 1,
                    "bi:%s:%s" % (bigram.prevtag, bigram.tag): 1,
                    }
.. code:: python

    data_X = map(lambda a: a.split(),
                 ["the dog walked",
                  "in the park",
                  "in the dog"])
    data_Y = map(lambda a: Bigram.from_tagging(a.split()),
                 ["D N V", "I D N", "I D N"])
.. code:: python

    def parse_training(handle):
        x = []
        y = []
        for l in handle:
            if not l.strip():
                yield (x, y)
                x = []
                y = []
            else:
                word, tag = l.split()
                x.append(word)
                y.append(tag)
        yield (x, y)
    data_X, data_Y = zip(*parse_training(open("tag/tag_train_small.dat")))
    data_Y = [Bigram.from_tagging(t) for t in data_Y] 
.. code:: python

    print data_Y[0]

.. parsed-literal::

    [Bigram(position=0, prevtag='<t>', tag='NOUN'), Bigram(position=1, prevtag='NOUN', tag='NOUN'), Bigram(position=2, prevtag='NOUN', tag='VERB'), Bigram(position=3, prevtag='VERB', tag='NOUN'), Bigram(position=4, prevtag='NOUN', tag='.'), Bigram(position=5, prevtag='.', tag='</t>')]


.. code:: python

    from pystruct.learners import StructuredPerceptron
    hm = TaggingCRFModel()
    sp = StructuredPerceptron(hm, verbose=1, max_iter=5)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        #fxn()
        sp.fit(data_X, data_Y)


.. parsed-literal::

    set(['NOUN', 'ADP', 'DET', '.', 'VERB', 'NUM', 'ADJ'])
    iteration 0
    Weights: [{}]
    SCORE IS: 0.0
    Bigram(position=-1, prevtag='<t>', tag='NOUN')
    Bigram(position=0, prevtag='NOUN', tag='NOUN')
    Bigram(position=1, prevtag='NOUN', tag='NOUN')
    Bigram(position=2, prevtag='NOUN', tag='NOUN')
    Bigram(position=3, prevtag='NOUN', tag='NOUN')
    Bigram(position=4, prevtag='NOUN', tag='</t>')
    {'word:tag:NOUN:Haag': 1, 'word:tag:NOUN:.': 1, 'bi:<t>:NOUN': 1, 'word:tag:NOUN:Ms.': 1, 'word:.': 2, 'word:Ms.': 1, 'word:plays': 1, 'bi:NOUN:</t>': 1, 'word:Elianti': 1, 'tag-1:NOUN': 5, 'bi:NOUN:NOUN': 4, 'word:tag:NOUN:Elianti': 1, 'word:tag:</t>:.': 1, 'word:Haag': 1, 'tag:</t>': 1, 'word:tag:NOUN:plays': 1, 'tag:NOUN': 5, 'tag-1:<t>': 1}
    {'word:tag:NOUN:Haag': 1, 'bi:<t>:NOUN': 1, 'word:tag:NOUN:Ms.': 1, 'word:Haag': 1, 'word:END': 1, 'word:Ms.': 1, 'tag-1:VERB': 1, 'word:tag:.:.': 1, 'bi:.:</t>': 1, 'word:Elianti': 1, 'word:tag:NOUN:Elianti': 1, 'tag:.': 1, 'tag:</t>': 1, 'tag:NOUN': 3, 'tag-1:<t>': 1, 'bi:NOUN:VERB': 1, 'tag:VERB': 1, 'word:tag:VERB:plays': 1, 'word:plays': 1, 'tag-1:NOUN': 3, 'bi:NOUN:NOUN': 1, 'word:tag:</t>:END': 1, 'bi:VERB:NOUN': 1, 'word:.': 1, 'bi:NOUN:.': 1, 'tag-1:.': 1}
    {'word:tag:NOUN:Haag': 1, 'word:tag:NOUN:.': 1, 'bi:<t>:NOUN': 1, 'word:tag:NOUN:Ms.': 1, 'word:.': 2, 'word:Ms.': 1, 'word:plays': 1, 'bi:NOUN:</t>': 1, 'word:Elianti': 1, 'tag-1:NOUN': 5, 'bi:NOUN:NOUN': 4, 'word:tag:NOUN:Elianti': 1, 'word:tag:</t>:.': 1, 'word:Haag': 1, 'tag:</t>': 1, 'word:tag:NOUN:plays': 1, 'tag:NOUN': 5, 'tag-1:<t>': 1}
    Weights: [{'bi:NOUN:VERB': 1.0, 'word:END': 1.0, 'word:tag:</t>:END': 1.0, 'tag:VERB': 1.0, 'word:.': -1.0, 'word:tag:VERB:plays': 1.0, 'bi:.:</t>': 1.0, 'bi:NOUN:.': 1.0, 'tag:NOUN': -2.0, 'tag-1:.': 1.0, 'tag-1:NOUN': -2.0, 'bi:NOUN:NOUN': -3.0, 'tag-1:VERB': 1.0, 'word:tag:.:.': 1.0, 'bi:VERB:NOUN': 1.0, 'bi:NOUN:</t>': -1.0, 'tag:.': 1.0}]
    SCORE IS: 25.0
    Bigram(position=-1, prevtag='<t>', tag='.')
    Bigram(position=0, prevtag='.', tag='.')
    Bigram(position=1, prevtag='.', tag='.')
    Bigram(position=2, prevtag='.', tag='.')
    Bigram(position=3, prevtag='.', tag='.')
    Bigram(position=4, prevtag='.', tag='.')
    Bigram(position=5, prevtag='.', tag='.')
    Bigram(position=6, prevtag='.', tag='.')
    Bigram(position=7, prevtag='.', tag='.')
    Bigram(position=8, prevtag='.', tag='.')
    Bigram(position=9, prevtag='.', tag='.')
    Bigram(position=10, prevtag='.', tag='.')
    Bigram(position=11, prevtag='.', tag='</t>')
    {'word:sold': 1, 'word:luxury': 1, 'word:tag:.:1,214': 1, 'word:tag:.:luxury': 1, 'word:last': 1, 'word:tag:.:the': 1, 'word:year': 1, 'bi:.:.': 11, 'word:tag:</t>:U.S.': 1, 'word:tag:.:auto': 1, 'word:The': 1, 'word:tag:.:cars': 1, 'word:1,214': 1, 'bi:.:</t>': 1, 'word:tag:.:maker': 1, 'word:U.S.': 2, 'word:tag:.:The': 1, 'word:tag:.:in': 1, 'tag:.': 12, 'tag:</t>': 1, 'tag-1:<t>': 1, 'word:tag:.:year': 1, 'word:maker': 1, 'word:in': 1, 'bi:<t>:.': 1, 'word:tag:.:last': 1, 'word:tag:.:sold': 1, 'tag-1:.': 12, 'word:auto': 1, 'word:cars': 1, 'word:the': 1, 'word:tag:.:U.S.': 1}
    {'word:sold': 1, 'word:luxury': 1, 'word:tag:VERB:sold': 1, 'word:U.S.': 1, 'word:END': 1, 'bi:NUM:NOUN': 1, 'word:last': 1, 'tag-1:VERB': 1, 'word:tag:NOUN:year': 1, 'word:year': 1, 'word:tag:NOUN:luxury': 1, 'bi:NOUN:</t>': 1, 'word:The': 1, 'word:1,214': 1, 'word:in': 1, 'tag:DET': 2, 'tag-1:DET': 2, 'bi:DET:NOUN': 2, 'bi:ADP:DET': 1, 'word:tag:NOUN:maker': 1, 'tag:</t>': 1, 'word:tag:NOUN:auto': 1, 'tag:NOUN': 6, 'tag-1:<t>': 1, 'bi:NOUN:VERB': 1, 'tag-1:NUM': 1, 'tag:VERB': 1, 'tag:ADJ': 1, 'bi:<t>:DET': 1, 'bi:VERB:NUM': 1, 'word:maker': 1, 'bi:NOUN:ADP': 1, 'tag-1:NOUN': 6, 'bi:NOUN:NOUN': 2, 'bi:NOUN:ADJ': 1, 'bi:ADJ:NOUN': 1, 'word:tag:DET:the': 1, 'word:tag:ADP:in': 1, 'tag:ADP': 1, 'word:tag:NOUN:cars': 1, 'tag:NUM': 1, 'word:tag:NOUN:U.S.': 1, 'tag-1:ADJ': 1, 'tag-1:ADP': 1, 'word:auto': 1, 'word:tag:NUM:1,214': 1, 'word:tag:DET:The': 1, 'word:cars': 1, 'word:the': 1, 'word:tag:ADJ:last': 1, 'word:tag:</t>:END': 1}
    {'word:sold': 1, 'word:luxury': 1, 'word:tag:.:1,214': 1, 'word:tag:.:luxury': 1, 'word:last': 1, 'word:tag:.:the': 1, 'word:year': 1, 'bi:.:.': 11, 'word:tag:</t>:U.S.': 1, 'word:tag:.:auto': 1, 'word:The': 1, 'word:tag:.:cars': 1, 'word:1,214': 1, 'bi:.:</t>': 1, 'word:tag:.:maker': 1, 'word:U.S.': 2, 'word:tag:.:The': 1, 'word:tag:.:in': 1, 'tag:.': 12, 'tag:</t>': 1, 'tag-1:<t>': 1, 'word:tag:.:year': 1, 'word:maker': 1, 'word:in': 1, 'bi:<t>:.': 1, 'word:tag:.:last': 1, 'word:tag:.:sold': 1, 'tag-1:.': 12, 'word:auto': 1, 'word:cars': 1, 'word:the': 1, 'word:tag:.:U.S.': 1}
    avg loss: 0.947368 w: [[  0.   1.   0.   1.   1.   2.   1.   0.   1.   1.  -1.   2.   1.   1.
        1. -11.   0.   1.   1.   2.   4.   1.   2. -11.   0.   1.   1.   2.
        4.   1.   2.  -1.   0.   2.   0.   0.   0.   0.  -1.   0.   0.   0.
        0.   0.   0.   0.   0.   1.   2.   1.   1.   1.   1.   0.   0.   0.
        1.   1.   1.   1.   1.   1.   1.   1.   1.   0.   0.]]
    effective learning rate: 1.000000
    iteration 1
    Weights: [{'word:END': 2.0, 'word:tag:VERB:sold': 1.0, 'word:tag:</t>:END': 2.0, 'bi:NUM:NOUN': 1.0, 'word:tag:ADP:in': 1.0, 'tag-1:ADJ': 1.0, 'word:tag:.:.': 1.0, 'word:tag:NOUN:year': 1.0, 'word:tag:NOUN:luxury': 1.0, 'word:U.S.': -1.0, 'tag-1:DET': 2.0, 'bi:DET:NOUN': 2.0, 'tag:.': -11.0, 'bi:ADP:DET': 1.0, 'word:tag:NOUN:maker': 1.0, 'word:tag:NOUN:auto': 1.0, 'tag:NOUN': 4.0, 'bi:NOUN:VERB': 2.0, 'tag-1:NUM': 1.0, 'word:tag:NUM:1,214': 1.0, 'tag:VERB': 2.0, 'tag:ADJ': 1.0, 'word:tag:VERB:plays': 1.0, 'bi:<t>:DET': 1.0, 'bi:VERB:NUM': 1.0, 'bi:NOUN:ADP': 1.0, 'tag-1:NOUN': 4.0, 'bi:NOUN:NOUN': -1.0, 'bi:NOUN:ADJ': 1.0, 'bi:ADJ:NOUN': 1.0, 'word:tag:DET:the': 1.0, 'bi:VERB:NOUN': 1.0, 'tag:ADP': 1.0, 'word:tag:NOUN:cars': 1.0, 'tag:NUM': 1.0, 'word:.': -1.0, 'word:tag:NOUN:U.S.': 1.0, 'bi:NOUN:.': 1.0, 'tag-1:.': -11.0, 'tag:DET': 2.0, 'tag-1:ADP': 1.0, 'word:tag:DET:The': 1.0, 'word:tag:ADJ:last': 1.0, 'tag-1:VERB': 2.0}]
    SCORE IS: 37.0
    Bigram(position=-1, prevtag='<t>', tag='NOUN')
    Bigram(position=0, prevtag='NOUN', tag='VERB')
    Bigram(position=1, prevtag='VERB', tag='NOUN')
    Bigram(position=2, prevtag='NOUN', tag='VERB')
    Bigram(position=3, prevtag='VERB', tag='NOUN')
    Bigram(position=4, prevtag='NOUN', tag='</t>')
    {'bi:NOUN:VERB': 2, 'word:tag:NOUN:Haag': 1, 'word:tag:NOUN:.': 1, 'bi:<t>:NOUN': 1, 'word:.': 2, 'word:Haag': 1, 'word:tag:VERB:plays': 1, 'word:Ms.': 1, 'word:plays': 1, 'bi:NOUN:</t>': 1, 'word:Elianti': 1, 'tag-1:NOUN': 3, 'word:tag:NOUN:Elianti': 1, 'tag-1:VERB': 2, 'tag:VERB': 2, 'word:tag:</t>:.': 1, 'bi:VERB:NOUN': 2, 'tag:</t>': 1, 'word:tag:VERB:Ms.': 1, 'tag:NOUN': 3, 'tag-1:<t>': 1}
    {'word:tag:NOUN:Haag': 1, 'bi:<t>:NOUN': 1, 'word:tag:NOUN:Ms.': 1, 'word:Haag': 1, 'word:END': 1, 'word:Ms.': 1, 'tag-1:VERB': 1, 'word:tag:.:.': 1, 'bi:.:</t>': 1, 'word:Elianti': 1, 'word:tag:NOUN:Elianti': 1, 'tag:.': 1, 'tag:</t>': 1, 'tag:NOUN': 3, 'tag-1:<t>': 1, 'bi:NOUN:VERB': 1, 'tag:VERB': 1, 'word:tag:VERB:plays': 1, 'word:plays': 1, 'tag-1:NOUN': 3, 'bi:NOUN:NOUN': 1, 'word:tag:</t>:END': 1, 'bi:VERB:NOUN': 1, 'word:.': 1, 'bi:NOUN:.': 1, 'tag-1:.': 1}
    {'bi:NOUN:VERB': 2, 'word:tag:NOUN:Haag': 1, 'word:tag:NOUN:.': 1, 'bi:<t>:NOUN': 1, 'word:.': 2, 'word:Haag': 1, 'word:tag:VERB:plays': 1, 'word:Ms.': 1, 'word:plays': 1, 'bi:NOUN:</t>': 1, 'word:Elianti': 1, 'tag-1:NOUN': 3, 'word:tag:NOUN:Elianti': 1, 'tag-1:VERB': 2, 'tag:VERB': 2, 'word:tag:</t>:.': 1, 'bi:VERB:NOUN': 2, 'tag:</t>': 1, 'word:tag:VERB:Ms.': 1, 'tag:NOUN': 3, 'tag-1:<t>': 1}
    Weights: [{'word:END': 3.0, 'word:tag:VERB:sold': 1.0, 'word:tag:NOUN:Ms.': 1.0, 'word:tag:</t>:END': 3.0, 'bi:DET:NOUN': 2.0, 'bi:NUM:NOUN': 1.0, 'word:tag:ADP:in': 1.0, 'tag-1:ADJ': 1.0, 'word:tag:.:.': 2.0, 'word:tag:NOUN:year': 1.0, 'word:tag:NOUN:luxury': 1.0, 'bi:NOUN:</t>': -1.0, 'word:U.S.': -1.0, 'tag-1:DET': 2.0, 'bi:.:</t>': 1.0, 'tag:.': -10.0, 'bi:ADP:DET': 1.0, 'word:tag:NOUN:maker': 1.0, 'word:tag:NOUN:auto': 1.0, 'tag:NOUN': 4.0, 'bi:NOUN:VERB': 1.0, 'tag-1:NUM': 1.0, 'word:tag:NUM:1,214': 1.0, 'tag:VERB': 1.0, 'tag:ADJ': 1.0, 'word:tag:VERB:plays': 1.0, 'bi:<t>:DET': 1.0, 'bi:VERB:NUM': 1.0, 'bi:NOUN:ADP': 1.0, 'tag-1:NOUN': 4.0, 'bi:NOUN:ADJ': 1.0, 'bi:ADJ:NOUN': 1.0, 'word:tag:DET:the': 1.0, 'tag:ADP': 1.0, 'word:tag:NOUN:cars': 1.0, 'tag:NUM': 1.0, 'word:.': -2.0, 'word:tag:NOUN:U.S.': 1.0, 'bi:NOUN:.': 2.0, 'tag-1:.': -10.0, 'tag:DET': 2.0, 'tag-1:ADP': 1.0, 'word:tag:DET:The': 1.0, 'word:tag:ADJ:last': 1.0, 'tag-1:VERB': 1.0}]
    SCORE IS: 99.0
    Bigram(position=-1, prevtag='<t>', tag='NOUN')
    Bigram(position=0, prevtag='NOUN', tag='NOUN')
    Bigram(position=1, prevtag='NOUN', tag='NOUN')
    Bigram(position=2, prevtag='NOUN', tag='NOUN')
    Bigram(position=3, prevtag='NOUN', tag='NOUN')
    Bigram(position=4, prevtag='NOUN', tag='NOUN')
    Bigram(position=5, prevtag='NOUN', tag='NOUN')
    Bigram(position=6, prevtag='NOUN', tag='NOUN')
    Bigram(position=7, prevtag='NOUN', tag='NOUN')
    Bigram(position=8, prevtag='NOUN', tag='NOUN')
    Bigram(position=9, prevtag='NOUN', tag='NOUN')
    Bigram(position=10, prevtag='NOUN', tag='NOUN')
    Bigram(position=11, prevtag='NOUN', tag='</t>')
    {'word:sold': 1, 'word:luxury': 1, 'bi:<t>:NOUN': 1, 'word:tag:NOUN:in': 1, 'word:tag:NOUN:the': 1, 'word:last': 1, 'word:tag:NOUN:last': 1, 'word:tag:NOUN:year': 1, 'word:year': 1, 'word:tag:NOUN:luxury': 1, 'bi:NOUN:</t>': 1, 'word:The': 1, 'word:1,214': 1, 'word:tag:NOUN:The': 1, 'word:U.S.': 2, 'word:tag:</t>:U.S.': 1, 'word:tag:NOUN:maker': 1, 'tag:</t>': 1, 'word:tag:NOUN:auto': 1, 'tag:NOUN': 12, 'tag-1:<t>': 1, 'word:maker': 1, 'tag-1:NOUN': 12, 'bi:NOUN:NOUN': 11, 'word:in': 1, 'word:tag:NOUN:cars': 1, 'word:tag:NOUN:sold': 1, 'word:tag:NOUN:1,214': 1, 'word:tag:NOUN:U.S.': 1, 'word:auto': 1, 'word:cars': 1, 'word:the': 1}
    {'word:sold': 1, 'word:luxury': 1, 'word:tag:VERB:sold': 1, 'word:U.S.': 1, 'word:END': 1, 'bi:NUM:NOUN': 1, 'word:last': 1, 'tag-1:VERB': 1, 'word:tag:NOUN:year': 1, 'word:year': 1, 'word:tag:NOUN:luxury': 1, 'bi:NOUN:</t>': 1, 'word:The': 1, 'word:1,214': 1, 'word:in': 1, 'tag:DET': 2, 'tag-1:DET': 2, 'bi:DET:NOUN': 2, 'bi:ADP:DET': 1, 'word:tag:NOUN:maker': 1, 'tag:</t>': 1, 'word:tag:NOUN:auto': 1, 'tag:NOUN': 6, 'tag-1:<t>': 1, 'bi:NOUN:VERB': 1, 'tag-1:NUM': 1, 'tag:VERB': 1, 'tag:ADJ': 1, 'bi:<t>:DET': 1, 'bi:VERB:NUM': 1, 'word:maker': 1, 'bi:NOUN:ADP': 1, 'tag-1:NOUN': 6, 'bi:NOUN:NOUN': 2, 'bi:NOUN:ADJ': 1, 'bi:ADJ:NOUN': 1, 'word:tag:DET:the': 1, 'word:tag:ADP:in': 1, 'tag:ADP': 1, 'word:tag:NOUN:cars': 1, 'tag:NUM': 1, 'word:tag:NOUN:U.S.': 1, 'tag-1:ADJ': 1, 'tag-1:ADP': 1, 'word:auto': 1, 'word:tag:NUM:1,214': 1, 'word:tag:DET:The': 1, 'word:cars': 1, 'word:the': 1, 'word:tag:ADJ:last': 1, 'word:tag:</t>:END': 1}
    {'word:sold': 1, 'word:luxury': 1, 'bi:<t>:NOUN': 1, 'word:tag:NOUN:in': 1, 'word:tag:NOUN:the': 1, 'word:last': 1, 'word:tag:NOUN:last': 1, 'word:tag:NOUN:year': 1, 'word:year': 1, 'word:tag:NOUN:luxury': 1, 'bi:NOUN:</t>': 1, 'word:The': 1, 'word:1,214': 1, 'word:tag:NOUN:The': 1, 'word:U.S.': 2, 'word:tag:</t>:U.S.': 1, 'word:tag:NOUN:maker': 1, 'tag:</t>': 1, 'word:tag:NOUN:auto': 1, 'tag:NOUN': 12, 'tag-1:<t>': 1, 'word:maker': 1, 'tag-1:NOUN': 12, 'bi:NOUN:NOUN': 11, 'word:in': 1, 'word:tag:NOUN:cars': 1, 'word:tag:NOUN:sold': 1, 'word:tag:NOUN:1,214': 1, 'word:tag:NOUN:U.S.': 1, 'word:auto': 1, 'word:cars': 1, 'word:the': 1}
    avg loss: 0.789474 w: [[  1.   2.  -1.   2.   2.   4.   2.  -1.   2.   2.  -9.   2.   2.   0.
        2. -10.   0.   2.   2.   4.  -2.   2.   2. -10.   0.   2.   2.   4.
       -2.   2.   2.  -2.   0.   4.   0.   0.   0.   0.  -2.   0.   0.   0.
        0.   0.   0.   0.   0.   2.   4.   2.   2.   2.   2.   0.   0.   1.
        1.   1.   1.   1.   1.   1.   2.   1.   2.   0.   0.]]
    effective learning rate: 1.000000
    iteration 2
    Weights: [{'word:END': 4.0, 'word:tag:VERB:sold': 2.0, 'bi:<t>:NOUN': -1.0, 'word:tag:NOUN:Ms.': 1.0, 'word:tag:</t>:END': 4.0, 'bi:DET:NOUN': 4.0, 'bi:NUM:NOUN': 2.0, 'word:tag:ADP:in': 2.0, 'tag-1:ADJ': 2.0, 'word:tag:.:.': 2.0, 'word:tag:NOUN:year': 1.0, 'word:tag:NOUN:luxury': 1.0, 'bi:NOUN:</t>': -1.0, 'word:U.S.': -2.0, 'tag-1:DET': 4.0, 'bi:.:</t>': 1.0, 'tag:.': -10.0, 'bi:ADP:DET': 2.0, 'word:tag:NOUN:maker': 1.0, 'word:tag:NOUN:auto': 1.0, 'tag:NOUN': -2.0, 'bi:NOUN:VERB': 2.0, 'tag-1:NUM': 2.0, 'word:tag:NUM:1,214': 2.0, 'tag:VERB': 2.0, 'tag:ADJ': 2.0, 'word:tag:VERB:plays': 1.0, 'bi:<t>:DET': 2.0, 'bi:VERB:NUM': 2.0, 'bi:NOUN:ADP': 2.0, 'tag-1:NOUN': -2.0, 'bi:NOUN:NOUN': -9.0, 'bi:NOUN:ADJ': 2.0, 'bi:ADJ:NOUN': 2.0, 'word:tag:DET:the': 2.0, 'tag:ADP': 2.0, 'word:tag:NOUN:cars': 1.0, 'tag:NUM': 2.0, 'word:.': -2.0, 'word:tag:NOUN:U.S.': 1.0, 'bi:NOUN:.': 2.0, 'tag-1:.': -10.0, 'tag:DET': 4.0, 'tag-1:ADP': 2.0, 'word:tag:DET:The': 2.0, 'word:tag:ADJ:last': 2.0, 'tag-1:VERB': 2.0}]
    SCORE IS: 38.0
    Bigram(position=-1, prevtag='<t>', tag='DET')
    Bigram(position=0, prevtag='DET', tag='DET')
    Bigram(position=1, prevtag='DET', tag='DET')
    Bigram(position=2, prevtag='DET', tag='DET')
    Bigram(position=3, prevtag='DET', tag='DET')
    Bigram(position=4, prevtag='DET', tag='</t>')
    {'bi:DET:</t>': 1, 'word:tag:DET:plays': 1, 'word:Haag': 1, 'word:.': 2, 'word:Ms.': 1, 'word:plays': 1, 'word:tag:DET:Elianti': 1, 'bi:DET:DET': 4, 'tag-1:DET': 5, 'bi:<t>:DET': 1, 'tag:DET': 5, 'word:tag:DET:Haag': 1, 'word:Elianti': 1, 'word:tag:DET:Ms.': 1, 'word:tag:DET:.': 1, 'tag:</t>': 1, 'word:tag:</t>:.': 1, 'tag-1:<t>': 1}
    {'word:tag:NOUN:Haag': 1, 'bi:<t>:NOUN': 1, 'word:tag:NOUN:Ms.': 1, 'word:Haag': 1, 'word:END': 1, 'word:Ms.': 1, 'tag-1:VERB': 1, 'word:tag:.:.': 1, 'bi:.:</t>': 1, 'word:Elianti': 1, 'word:tag:NOUN:Elianti': 1, 'tag:.': 1, 'tag:</t>': 1, 'tag:NOUN': 3, 'tag-1:<t>': 1, 'bi:NOUN:VERB': 1, 'tag:VERB': 1, 'word:tag:VERB:plays': 1, 'word:plays': 1, 'tag-1:NOUN': 3, 'bi:NOUN:NOUN': 1, 'word:tag:</t>:END': 1, 'bi:VERB:NOUN': 1, 'word:.': 1, 'bi:NOUN:.': 1, 'tag-1:.': 1}
    {'bi:DET:</t>': 1, 'word:tag:DET:plays': 1, 'word:Haag': 1, 'word:.': 2, 'word:Ms.': 1, 'word:plays': 1, 'word:tag:DET:Elianti': 1, 'bi:DET:DET': 4, 'tag-1:DET': 5, 'bi:<t>:DET': 1, 'tag:DET': 5, 'word:tag:DET:Haag': 1, 'word:Elianti': 1, 'word:tag:DET:Ms.': 1, 'word:tag:DET:.': 1, 'tag:</t>': 1, 'word:tag:</t>:.': 1, 'tag-1:<t>': 1}
    Weights: [{'word:END': 5.0, 'word:tag:VERB:sold': 2.0, 'word:tag:NOUN:Ms.': 2.0, 'word:tag:</t>:END': 5.0, 'word:tag:NOUN:Haag': 1.0, 'bi:DET:NOUN': 4.0, 'bi:NUM:NOUN': 2.0, 'word:tag:ADP:in': 2.0, 'tag-1:ADJ': 2.0, 'word:tag:.:.': 3.0, 'word:tag:NOUN:year': 1.0, 'word:tag:NOUN:luxury': 1.0, 'bi:NOUN:</t>': -1.0, 'word:U.S.': -2.0, 'tag-1:DET': -1.0, 'word:tag:NOUN:Elianti': 1.0, 'bi:.:</t>': 2.0, 'tag:.': -9.0, 'bi:ADP:DET': 2.0, 'word:tag:NOUN:maker': 1.0, 'word:tag:NOUN:auto': 1.0, 'tag:NOUN': 1.0, 'bi:NOUN:VERB': 3.0, 'tag-1:NUM': 2.0, 'word:tag:NUM:1,214': 2.0, 'tag:VERB': 3.0, 'tag:ADJ': 2.0, 'word:tag:VERB:plays': 2.0, 'bi:<t>:DET': 1.0, 'bi:VERB:NUM': 2.0, 'bi:NOUN:ADP': 2.0, 'tag-1:NOUN': 1.0, 'bi:NOUN:NOUN': -8.0, 'bi:NOUN:ADJ': 2.0, 'bi:ADJ:NOUN': 2.0, 'word:tag:DET:the': 2.0, 'bi:VERB:NOUN': 1.0, 'tag:ADP': 2.0, 'word:tag:NOUN:cars': 1.0, 'tag:NUM': 2.0, 'word:.': -3.0, 'word:tag:NOUN:U.S.': 1.0, 'bi:NOUN:.': 3.0, 'tag-1:.': -9.0, 'tag:DET': -1.0, 'tag-1:ADP': 2.0, 'word:tag:DET:The': 2.0, 'word:tag:ADJ:last': 2.0, 'tag-1:VERB': 3.0}]
    SCORE IS: 78.0
    Bigram(position=-1, prevtag='<t>', tag='VERB')
    Bigram(position=0, prevtag='VERB', tag='NUM')
    Bigram(position=1, prevtag='NUM', tag='NOUN')
    Bigram(position=2, prevtag='NOUN', tag='VERB')
    Bigram(position=3, prevtag='VERB', tag='NOUN')
    Bigram(position=4, prevtag='NOUN', tag='VERB')
    Bigram(position=5, prevtag='VERB', tag='NOUN')
    Bigram(position=6, prevtag='NOUN', tag='VERB')
    Bigram(position=7, prevtag='VERB', tag='NUM')
    Bigram(position=8, prevtag='NUM', tag='NOUN')
    Bigram(position=9, prevtag='NOUN', tag='VERB')
    Bigram(position=10, prevtag='VERB', tag='VERB')
    Bigram(position=11, prevtag='VERB', tag='</t>')
    {'word:sold': 1, 'word:luxury': 1, 'word:tag:VERB:sold': 1, 'bi:NUM:NOUN': 2, 'word:tag:NOUN:year': 1, 'word:last': 1, 'tag-1:VERB': 6, 'bi:<t>:VERB': 1, 'word:year': 1, 'word:tag:NOUN:luxury': 1, 'word:tag:NUM:The': 1, 'word:The': 1, 'word:tag:VERB:in': 1, 'word:1,214': 1, 'word:U.S.': 2, 'word:tag:VERB:the': 1, 'word:tag:</t>:U.S.': 1, 'word:tag:NOUN:maker': 1, 'tag:</t>': 1, 'word:tag:VERB:auto': 1, 'word:tag:VERB:last': 1, 'word:the': 1, 'bi:NOUN:VERB': 4, 'tag-1:NUM': 2, 'tag:VERB': 6, 'bi:VERB:NUM': 2, 'word:maker': 1, 'tag-1:NOUN': 4, 'bi:VERB:NOUN': 2, 'word:in': 1, 'word:tag:NOUN:cars': 1, 'bi:VERB:</t>': 1, 'bi:VERB:VERB': 1, 'tag:NUM': 2, 'word:auto': 1, 'word:tag:VERB:U.S.': 1, 'word:tag:NUM:1,214': 1, 'tag:NOUN': 4, 'word:cars': 1, 'tag-1:<t>': 1}
    {'word:sold': 1, 'word:luxury': 1, 'word:tag:VERB:sold': 1, 'word:U.S.': 1, 'word:END': 1, 'bi:NUM:NOUN': 1, 'word:last': 1, 'tag-1:VERB': 1, 'word:tag:NOUN:year': 1, 'word:year': 1, 'word:tag:NOUN:luxury': 1, 'bi:NOUN:</t>': 1, 'word:The': 1, 'word:1,214': 1, 'word:in': 1, 'tag:DET': 2, 'tag-1:DET': 2, 'bi:DET:NOUN': 2, 'bi:ADP:DET': 1, 'word:tag:NOUN:maker': 1, 'tag:</t>': 1, 'word:tag:NOUN:auto': 1, 'tag:NOUN': 6, 'tag-1:<t>': 1, 'bi:NOUN:VERB': 1, 'tag-1:NUM': 1, 'tag:VERB': 1, 'tag:ADJ': 1, 'bi:<t>:DET': 1, 'bi:VERB:NUM': 1, 'word:maker': 1, 'bi:NOUN:ADP': 1, 'tag-1:NOUN': 6, 'bi:NOUN:NOUN': 2, 'bi:NOUN:ADJ': 1, 'bi:ADJ:NOUN': 1, 'word:tag:DET:the': 1, 'word:tag:ADP:in': 1, 'tag:ADP': 1, 'word:tag:NOUN:cars': 1, 'tag:NUM': 1, 'word:tag:NOUN:U.S.': 1, 'tag-1:ADJ': 1, 'tag-1:ADP': 1, 'word:auto': 1, 'word:tag:NUM:1,214': 1, 'word:tag:DET:The': 1, 'word:cars': 1, 'word:the': 1, 'word:tag:ADJ:last': 1, 'word:tag:</t>:END': 1}
    {'word:sold': 1, 'word:luxury': 1, 'word:tag:VERB:sold': 1, 'bi:NUM:NOUN': 2, 'word:tag:NOUN:year': 1, 'word:last': 1, 'tag-1:VERB': 6, 'bi:<t>:VERB': 1, 'word:year': 1, 'word:tag:NOUN:luxury': 1, 'word:tag:NUM:The': 1, 'word:The': 1, 'word:tag:VERB:in': 1, 'word:1,214': 1, 'word:U.S.': 2, 'word:tag:VERB:the': 1, 'word:tag:</t>:U.S.': 1, 'word:tag:NOUN:maker': 1, 'tag:</t>': 1, 'word:tag:VERB:auto': 1, 'word:tag:VERB:last': 1, 'word:the': 1, 'bi:NOUN:VERB': 4, 'tag-1:NUM': 2, 'tag:VERB': 6, 'bi:VERB:NUM': 2, 'word:maker': 1, 'tag-1:NOUN': 4, 'bi:VERB:NOUN': 2, 'word:in': 1, 'word:tag:NOUN:cars': 1, 'bi:VERB:</t>': 1, 'bi:VERB:VERB': 1, 'tag:NUM': 2, 'word:auto': 1, 'word:tag:VERB:U.S.': 1, 'word:tag:NUM:1,214': 1, 'tag:NOUN': 4, 'word:cars': 1, 'tag-1:<t>': 1}
    avg loss: 0.842105 w: [[ 2.  2.  0.  3.  3.  6.  3.  0.  3.  3. -6.  0.  1. -1.  1. -9.  0.  3.
       3.  1.  3.  1. -2. -9.  0.  3.  3.  1.  3.  1. -2. -3.  0.  6.  0.  0.
       0.  0. -3.  0.  0.  0.  0.  0.  0.  0.  0.  3.  6.  3.  3.  3.  3.  1.
       1.  2.  2.  2.  1.  1.  1.  1.  2.  2.  2.  0.  0.]]
    effective learning rate: 1.000000
    iteration 3
    Weights: [{'word:END': 6.0, 'word:tag:VERB:sold': 2.0, 'word:tag:NOUN:Ms.': 2.0, 'word:tag:</t>:END': 6.0, 'word:tag:NOUN:Haag': 1.0, 'bi:DET:NOUN': 6.0, 'bi:NUM:NOUN': 1.0, 'word:tag:ADP:in': 3.0, 'tag-1:ADJ': 3.0, 'word:tag:.:.': 3.0, 'word:tag:NOUN:year': 1.0, 'word:tag:NOUN:luxury': 1.0, 'word:U.S.': -3.0, 'tag-1:DET': 1.0, 'word:tag:NOUN:Elianti': 1.0, 'bi:.:</t>': 2.0, 'tag:.': -9.0, 'bi:ADP:DET': 3.0, 'word:tag:NOUN:maker': 1.0, 'word:tag:NOUN:auto': 2.0, 'tag:NOUN': 3.0, 'tag-1:NUM': 1.0, 'word:tag:NUM:1,214': 2.0, 'tag:VERB': -2.0, 'tag:ADJ': 3.0, 'word:tag:VERB:plays': 2.0, 'bi:<t>:DET': 2.0, 'bi:VERB:NUM': 1.0, 'bi:NOUN:ADP': 3.0, 'tag-1:NOUN': 3.0, 'bi:NOUN:NOUN': -6.0, 'bi:NOUN:ADJ': 3.0, 'bi:ADJ:NOUN': 3.0, 'word:tag:DET:the': 3.0, 'bi:VERB:NOUN': -1.0, 'tag:ADP': 3.0, 'word:tag:NOUN:cars': 1.0, 'tag:NUM': 1.0, 'word:.': -3.0, 'word:tag:NOUN:U.S.': 2.0, 'bi:NOUN:.': 3.0, 'tag-1:.': -9.0, 'tag:DET': 1.0, 'tag-1:ADP': 3.0, 'word:tag:DET:The': 3.0, 'word:tag:ADJ:last': 3.0, 'tag-1:VERB': -2.0}]
    SCORE IS: 39.0
    Bigram(position=-1, prevtag='<t>', tag='DET')
    Bigram(position=0, prevtag='DET', tag='NOUN')
    Bigram(position=1, prevtag='NOUN', tag='ADP')
    Bigram(position=2, prevtag='ADP', tag='DET')
    Bigram(position=3, prevtag='DET', tag='NOUN')
    Bigram(position=4, prevtag='NOUN', tag='</t>')
    {'word:tag:NOUN:Ms.': 1, 'word:Haag': 1, 'word:Ms.': 1, 'word:Elianti': 1, 'word:tag:DET:.': 1, 'bi:NOUN:</t>': 1, 'tag-1:DET': 2, 'bi:DET:NOUN': 2, 'word:tag:NOUN:Elianti': 1, 'bi:ADP:DET': 1, 'tag:</t>': 1, 'tag:NOUN': 2, 'tag-1:<t>': 1, 'bi:<t>:DET': 1, 'word:plays': 1, 'bi:NOUN:ADP': 1, 'tag-1:NOUN': 2, 'tag:ADP': 1, 'word:tag:ADP:Haag': 1, 'word:.': 2, 'tag:DET': 2, 'tag-1:ADP': 1, 'word:tag:DET:plays': 1, 'word:tag:</t>:.': 1}
    {'word:tag:NOUN:Haag': 1, 'bi:<t>:NOUN': 1, 'word:tag:NOUN:Ms.': 1, 'word:Haag': 1, 'word:END': 1, 'word:Ms.': 1, 'tag-1:VERB': 1, 'word:tag:.:.': 1, 'bi:.:</t>': 1, 'word:Elianti': 1, 'word:tag:NOUN:Elianti': 1, 'tag:.': 1, 'tag:</t>': 1, 'tag:NOUN': 3, 'tag-1:<t>': 1, 'bi:NOUN:VERB': 1, 'tag:VERB': 1, 'word:tag:VERB:plays': 1, 'word:plays': 1, 'tag-1:NOUN': 3, 'bi:NOUN:NOUN': 1, 'word:tag:</t>:END': 1, 'bi:VERB:NOUN': 1, 'word:.': 1, 'bi:NOUN:.': 1, 'tag-1:.': 1}
    {'word:tag:NOUN:Ms.': 1, 'word:Haag': 1, 'word:Ms.': 1, 'word:Elianti': 1, 'word:tag:DET:.': 1, 'bi:NOUN:</t>': 1, 'tag-1:DET': 2, 'bi:DET:NOUN': 2, 'word:tag:NOUN:Elianti': 1, 'bi:ADP:DET': 1, 'tag:</t>': 1, 'tag:NOUN': 2, 'tag-1:<t>': 1, 'bi:<t>:DET': 1, 'word:plays': 1, 'bi:NOUN:ADP': 1, 'tag-1:NOUN': 2, 'tag:ADP': 1, 'word:tag:ADP:Haag': 1, 'word:.': 2, 'tag:DET': 2, 'tag-1:ADP': 1, 'word:tag:DET:plays': 1, 'word:tag:</t>:.': 1}
    Weights: [{'word:END': 7.0, 'word:tag:VERB:sold': 2.0, 'bi:<t>:NOUN': 1.0, 'word:tag:NOUN:Ms.': 2.0, 'word:tag:</t>:END': 7.0, 'word:tag:NOUN:Haag': 2.0, 'bi:DET:NOUN': 4.0, 'bi:NUM:NOUN': 1.0, 'word:tag:ADP:in': 3.0, 'tag-1:ADJ': 3.0, 'word:tag:.:.': 4.0, 'word:tag:NOUN:year': 1.0, 'word:tag:NOUN:luxury': 1.0, 'bi:NOUN:</t>': -1.0, 'word:U.S.': -3.0, 'tag-1:DET': -1.0, 'word:tag:NOUN:Elianti': 1.0, 'bi:.:</t>': 3.0, 'tag:.': -8.0, 'bi:ADP:DET': 2.0, 'word:tag:NOUN:maker': 1.0, 'word:tag:NOUN:auto': 2.0, 'tag:NOUN': 4.0, 'bi:NOUN:VERB': 1.0, 'tag-1:NUM': 1.0, 'word:tag:NUM:1,214': 2.0, 'tag:VERB': -1.0, 'tag:ADJ': 3.0, 'word:tag:VERB:plays': 3.0, 'bi:<t>:DET': 1.0, 'bi:VERB:NUM': 1.0, 'bi:NOUN:ADP': 2.0, 'tag-1:NOUN': 4.0, 'bi:NOUN:NOUN': -5.0, 'bi:NOUN:ADJ': 3.0, 'bi:ADJ:NOUN': 3.0, 'word:tag:DET:the': 3.0, 'tag:ADP': 2.0, 'word:tag:NOUN:cars': 1.0, 'tag:NUM': 1.0, 'word:.': -4.0, 'word:tag:NOUN:U.S.': 2.0, 'bi:NOUN:.': 4.0, 'tag-1:.': -8.0, 'tag:DET': -1.0, 'tag-1:ADP': 2.0, 'word:tag:DET:The': 3.0, 'word:tag:ADJ:last': 3.0, 'tag-1:VERB': -1.0}]
    SCORE IS: 120.0
    Bigram(position=-1, prevtag='<t>', tag='NOUN')
    Bigram(position=0, prevtag='NOUN', tag='ADJ')
    Bigram(position=1, prevtag='ADJ', tag='NOUN')
    Bigram(position=2, prevtag='NOUN', tag='ADJ')
    Bigram(position=3, prevtag='ADJ', tag='NOUN')
    Bigram(position=4, prevtag='NOUN', tag='ADJ')
    Bigram(position=5, prevtag='ADJ', tag='NOUN')
    Bigram(position=6, prevtag='NOUN', tag='ADJ')
    Bigram(position=7, prevtag='ADJ', tag='NOUN')
    Bigram(position=8, prevtag='NOUN', tag='ADJ')
    Bigram(position=9, prevtag='ADJ', tag='NOUN')
    Bigram(position=10, prevtag='NOUN', tag='ADJ')
    Bigram(position=11, prevtag='ADJ', tag='</t>')
    {'word:sold': 1, 'word:luxury': 1, 'bi:<t>:NOUN': 1, 'word:tag:ADJ:auto': 1, 'word:last': 1, 'word:tag:ADJ:The': 1, 'word:tag:NOUN:year': 1, 'word:year': 1, 'word:tag:NOUN:luxury': 1, 'word:The': 1, 'word:1,214': 1, 'word:U.S.': 2, 'word:tag:</t>:U.S.': 1, 'bi:ADJ:</t>': 1, 'word:tag:ADJ:the': 1, 'word:tag:NOUN:maker': 1, 'word:tag:NOUN:in': 1, 'word:the': 1, 'tag:NOUN': 6, 'tag-1:<t>': 1, 'tag:ADJ': 6, 'word:maker': 1, 'tag-1:NOUN': 6, 'bi:NOUN:ADJ': 6, 'bi:ADJ:NOUN': 5, 'word:in': 1, 'tag:</t>': 1, 'word:tag:NOUN:1,214': 1, 'word:tag:ADJ:sold': 1, 'word:tag:NOUN:U.S.': 1, 'tag-1:ADJ': 6, 'word:auto': 1, 'word:cars': 1, 'word:tag:ADJ:cars': 1, 'word:tag:ADJ:last': 1}
    {'word:sold': 1, 'word:luxury': 1, 'word:tag:VERB:sold': 1, 'word:U.S.': 1, 'word:END': 1, 'bi:NUM:NOUN': 1, 'word:last': 1, 'tag-1:VERB': 1, 'word:tag:NOUN:year': 1, 'word:year': 1, 'word:tag:NOUN:luxury': 1, 'bi:NOUN:</t>': 1, 'word:The': 1, 'word:1,214': 1, 'word:in': 1, 'tag:DET': 2, 'tag-1:DET': 2, 'bi:DET:NOUN': 2, 'bi:ADP:DET': 1, 'word:tag:NOUN:maker': 1, 'tag:</t>': 1, 'word:tag:NOUN:auto': 1, 'tag:NOUN': 6, 'tag-1:<t>': 1, 'bi:NOUN:VERB': 1, 'tag-1:NUM': 1, 'tag:VERB': 1, 'tag:ADJ': 1, 'bi:<t>:DET': 1, 'bi:VERB:NUM': 1, 'word:maker': 1, 'bi:NOUN:ADP': 1, 'tag-1:NOUN': 6, 'bi:NOUN:NOUN': 2, 'bi:NOUN:ADJ': 1, 'bi:ADJ:NOUN': 1, 'word:tag:DET:the': 1, 'word:tag:ADP:in': 1, 'tag:ADP': 1, 'word:tag:NOUN:cars': 1, 'tag:NUM': 1, 'word:tag:NOUN:U.S.': 1, 'tag-1:ADJ': 1, 'tag-1:ADP': 1, 'word:auto': 1, 'word:tag:NUM:1,214': 1, 'word:tag:DET:The': 1, 'word:cars': 1, 'word:the': 1, 'word:tag:ADJ:last': 1, 'word:tag:</t>:END': 1}
    {'word:sold': 1, 'word:luxury': 1, 'bi:<t>:NOUN': 1, 'word:tag:ADJ:auto': 1, 'word:last': 1, 'word:tag:ADJ:The': 1, 'word:tag:NOUN:year': 1, 'word:year': 1, 'word:tag:NOUN:luxury': 1, 'word:The': 1, 'word:1,214': 1, 'word:U.S.': 2, 'word:tag:</t>:U.S.': 1, 'bi:ADJ:</t>': 1, 'word:tag:ADJ:the': 1, 'word:tag:NOUN:maker': 1, 'word:tag:NOUN:in': 1, 'word:the': 1, 'tag:NOUN': 6, 'tag-1:<t>': 1, 'tag:ADJ': 6, 'word:maker': 1, 'tag-1:NOUN': 6, 'bi:NOUN:ADJ': 6, 'bi:ADJ:NOUN': 5, 'word:in': 1, 'tag:</t>': 1, 'word:tag:NOUN:1,214': 1, 'word:tag:ADJ:sold': 1, 'word:tag:NOUN:U.S.': 1, 'tag-1:ADJ': 6, 'word:auto': 1, 'word:cars': 1, 'word:tag:ADJ:cars': 1, 'word:tag:ADJ:last': 1}
    avg loss: 0.894737 w: [[ 3.  2.  0. -1.  3.  6.  4.  0. -2.  3. -3.  2.  2.  0.  2. -8.  0. -2.
       3.  1.  4.  2.  0. -8.  0. -2.  3.  1.  4.  2.  0. -4.  0.  8.  0.  0.
       0.  0. -4.  0.  0.  0.  0.  0.  0.  0.  0.  4.  8.  3.  4.  4.  4.  1.
       2.  2.  2.  3.  2.  1.  1.  1.  3.  3.  3.  0.  0.]]
    effective learning rate: 1.000000
    iteration 4
    Weights: [{'word:END': 8.0, 'word:tag:VERB:sold': 3.0, 'word:tag:NOUN:Ms.': 2.0, 'word:tag:NOUN:Haag': 2.0, 'bi:DET:NOUN': 6.0, 'bi:NUM:NOUN': 2.0, 'word:tag:ADP:in': 4.0, 'tag-1:ADJ': -2.0, 'word:tag:.:.': 4.0, 'word:tag:NOUN:year': 1.0, 'word:tag:NOUN:luxury': 1.0, 'word:U.S.': -4.0, 'tag-1:DET': 1.0, 'word:tag:NOUN:Elianti': 1.0, 'bi:.:</t>': 3.0, 'tag:.': -8.0, 'bi:ADP:DET': 3.0, 'word:tag:NOUN:maker': 1.0, 'word:tag:NOUN:auto': 3.0, 'tag:NOUN': 4.0, 'bi:NOUN:VERB': 2.0, 'tag-1:NUM': 2.0, 'word:tag:NUM:1,214': 3.0, 'tag:ADJ': -2.0, 'word:tag:VERB:plays': 3.0, 'bi:<t>:DET': 2.0, 'bi:VERB:NUM': 2.0, 'bi:NOUN:ADP': 3.0, 'tag-1:NOUN': 4.0, 'bi:NOUN:NOUN': -3.0, 'bi:NOUN:ADJ': -2.0, 'bi:ADJ:NOUN': -1.0, 'word:tag:DET:the': 4.0, 'tag:ADP': 3.0, 'word:tag:NOUN:cars': 2.0, 'tag:NUM': 2.0, 'word:.': -4.0, 'word:tag:NOUN:U.S.': 2.0, 'bi:NOUN:.': 4.0, 'tag-1:.': -8.0, 'tag:DET': 1.0, 'tag-1:ADP': 3.0, 'word:tag:DET:The': 4.0, 'word:tag:ADJ:last': 3.0, 'word:tag:</t>:END': 8.0}]
    SCORE IS: 41.0
    Bigram(position=-1, prevtag='<t>', tag='DET')
    Bigram(position=0, prevtag='DET', tag='NOUN')
    Bigram(position=1, prevtag='NOUN', tag='ADP')
    Bigram(position=2, prevtag='ADP', tag='DET')
    Bigram(position=3, prevtag='DET', tag='NOUN')
    Bigram(position=4, prevtag='NOUN', tag='</t>')
    {'word:tag:NOUN:Ms.': 1, 'word:Haag': 1, 'word:Ms.': 1, 'word:Elianti': 1, 'word:tag:DET:.': 1, 'bi:NOUN:</t>': 1, 'tag-1:DET': 2, 'bi:DET:NOUN': 2, 'word:tag:NOUN:Elianti': 1, 'bi:ADP:DET': 1, 'tag:</t>': 1, 'tag:NOUN': 2, 'tag-1:<t>': 1, 'bi:<t>:DET': 1, 'word:plays': 1, 'bi:NOUN:ADP': 1, 'tag-1:NOUN': 2, 'tag:ADP': 1, 'word:tag:ADP:Haag': 1, 'word:.': 2, 'tag:DET': 2, 'tag-1:ADP': 1, 'word:tag:DET:plays': 1, 'word:tag:</t>:.': 1}
    {'word:tag:NOUN:Haag': 1, 'bi:<t>:NOUN': 1, 'word:tag:NOUN:Ms.': 1, 'word:Haag': 1, 'word:END': 1, 'word:Ms.': 1, 'tag-1:VERB': 1, 'word:tag:.:.': 1, 'bi:.:</t>': 1, 'word:Elianti': 1, 'word:tag:NOUN:Elianti': 1, 'tag:.': 1, 'tag:</t>': 1, 'tag:NOUN': 3, 'tag-1:<t>': 1, 'bi:NOUN:VERB': 1, 'tag:VERB': 1, 'word:tag:VERB:plays': 1, 'word:plays': 1, 'tag-1:NOUN': 3, 'bi:NOUN:NOUN': 1, 'word:tag:</t>:END': 1, 'bi:VERB:NOUN': 1, 'word:.': 1, 'bi:NOUN:.': 1, 'tag-1:.': 1}
    {'word:tag:NOUN:Ms.': 1, 'word:Haag': 1, 'word:Ms.': 1, 'word:Elianti': 1, 'word:tag:DET:.': 1, 'bi:NOUN:</t>': 1, 'tag-1:DET': 2, 'bi:DET:NOUN': 2, 'word:tag:NOUN:Elianti': 1, 'bi:ADP:DET': 1, 'tag:</t>': 1, 'tag:NOUN': 2, 'tag-1:<t>': 1, 'bi:<t>:DET': 1, 'word:plays': 1, 'bi:NOUN:ADP': 1, 'tag-1:NOUN': 2, 'tag:ADP': 1, 'word:tag:ADP:Haag': 1, 'word:.': 2, 'tag:DET': 2, 'tag-1:ADP': 1, 'word:tag:DET:plays': 1, 'word:tag:</t>:.': 1}
    Weights: [{'word:END': 9.0, 'word:tag:VERB:sold': 3.0, 'bi:<t>:NOUN': 1.0, 'word:tag:NOUN:Ms.': 2.0, 'word:tag:</t>:END': 9.0, 'word:tag:NOUN:Haag': 3.0, 'bi:DET:NOUN': 4.0, 'bi:NUM:NOUN': 2.0, 'word:tag:ADP:in': 4.0, 'tag-1:ADJ': -2.0, 'word:tag:.:.': 5.0, 'word:tag:NOUN:year': 1.0, 'word:tag:NOUN:luxury': 1.0, 'bi:NOUN:</t>': -1.0, 'word:U.S.': -4.0, 'tag-1:DET': -1.0, 'word:tag:NOUN:Elianti': 1.0, 'bi:.:</t>': 4.0, 'tag:.': -7.0, 'bi:ADP:DET': 2.0, 'word:tag:NOUN:maker': 1.0, 'word:tag:NOUN:auto': 3.0, 'tag:NOUN': 5.0, 'bi:NOUN:VERB': 3.0, 'tag-1:NUM': 2.0, 'word:tag:NUM:1,214': 3.0, 'tag:VERB': 1.0, 'tag:ADJ': -2.0, 'word:tag:VERB:plays': 4.0, 'bi:<t>:DET': 1.0, 'bi:VERB:NUM': 2.0, 'bi:NOUN:ADP': 2.0, 'tag-1:NOUN': 5.0, 'bi:NOUN:NOUN': -2.0, 'bi:NOUN:ADJ': -2.0, 'bi:ADJ:NOUN': -1.0, 'word:tag:DET:the': 4.0, 'bi:VERB:NOUN': 1.0, 'tag:ADP': 2.0, 'word:tag:NOUN:cars': 2.0, 'tag:NUM': 2.0, 'word:.': -5.0, 'word:tag:NOUN:U.S.': 2.0, 'bi:NOUN:.': 5.0, 'tag-1:.': -7.0, 'tag:DET': -1.0, 'tag-1:ADP': 2.0, 'word:tag:DET:The': 4.0, 'word:tag:ADJ:last': 3.0, 'tag-1:VERB': 1.0}]
    SCORE IS: 109.0
    Bigram(position=-1, prevtag='<t>', tag='NOUN')
    Bigram(position=0, prevtag='NOUN', tag='NOUN')
    Bigram(position=1, prevtag='NOUN', tag='NOUN')
    Bigram(position=2, prevtag='NOUN', tag='NOUN')
    Bigram(position=3, prevtag='NOUN', tag='NOUN')
    Bigram(position=4, prevtag='NOUN', tag='NOUN')
    Bigram(position=5, prevtag='NOUN', tag='NOUN')
    Bigram(position=6, prevtag='NOUN', tag='VERB')
    Bigram(position=7, prevtag='VERB', tag='NUM')
    Bigram(position=8, prevtag='NUM', tag='NOUN')
    Bigram(position=9, prevtag='NOUN', tag='ADP')
    Bigram(position=10, prevtag='ADP', tag='NOUN')
    Bigram(position=11, prevtag='NOUN', tag='</t>')
    {'word:sold': 1, 'word:luxury': 1, 'word:tag:VERB:sold': 1, 'bi:<t>:NOUN': 1, 'bi:NUM:NOUN': 1, 'word:tag:ADP:in': 1, 'word:tag:NOUN:last': 1, 'tag-1:VERB': 1, 'word:tag:NOUN:year': 1, 'word:year': 1, 'word:tag:NOUN:luxury': 1, 'bi:NOUN:</t>': 1, 'word:The': 1, 'word:1,214': 1, 'word:in': 1, 'word:tag:NOUN:The': 1, 'word:U.S.': 2, 'word:tag:</t>:U.S.': 1, 'word:tag:NOUN:maker': 1, 'tag:</t>': 1, 'word:tag:NOUN:the': 1, 'tag:NOUN': 9, 'tag-1:<t>': 1, 'bi:NOUN:VERB': 1, 'tag-1:NUM': 1, 'tag:VERB': 1, 'bi:VERB:NUM': 1, 'word:maker': 1, 'bi:NOUN:ADP': 1, 'tag-1:NOUN': 9, 'bi:NOUN:NOUN': 6, 'word:last': 1, 'tag:ADP': 1, 'word:tag:NOUN:cars': 1, 'tag:NUM': 1, 'word:tag:NOUN:U.S.': 1, 'word:tag:NOUN:auto': 1, 'tag-1:ADP': 1, 'word:auto': 1, 'word:tag:NUM:1,214': 1, 'word:cars': 1, 'word:the': 1, 'bi:ADP:NOUN': 1}
    {'word:sold': 1, 'word:luxury': 1, 'word:tag:VERB:sold': 1, 'word:U.S.': 1, 'word:END': 1, 'bi:NUM:NOUN': 1, 'word:last': 1, 'tag-1:VERB': 1, 'word:tag:NOUN:year': 1, 'word:year': 1, 'word:tag:NOUN:luxury': 1, 'bi:NOUN:</t>': 1, 'word:The': 1, 'word:1,214': 1, 'word:in': 1, 'tag:DET': 2, 'tag-1:DET': 2, 'bi:DET:NOUN': 2, 'bi:ADP:DET': 1, 'word:tag:NOUN:maker': 1, 'tag:</t>': 1, 'word:tag:NOUN:auto': 1, 'tag:NOUN': 6, 'tag-1:<t>': 1, 'bi:NOUN:VERB': 1, 'tag-1:NUM': 1, 'tag:VERB': 1, 'tag:ADJ': 1, 'bi:<t>:DET': 1, 'bi:VERB:NUM': 1, 'word:maker': 1, 'bi:NOUN:ADP': 1, 'tag-1:NOUN': 6, 'bi:NOUN:NOUN': 2, 'bi:NOUN:ADJ': 1, 'bi:ADJ:NOUN': 1, 'word:tag:DET:the': 1, 'word:tag:ADP:in': 1, 'tag:ADP': 1, 'word:tag:NOUN:cars': 1, 'tag:NUM': 1, 'word:tag:NOUN:U.S.': 1, 'tag-1:ADJ': 1, 'tag-1:ADP': 1, 'word:auto': 1, 'word:tag:NUM:1,214': 1, 'word:tag:DET:The': 1, 'word:cars': 1, 'word:the': 1, 'word:tag:ADJ:last': 1, 'word:tag:</t>:END': 1}
    {'word:sold': 1, 'word:luxury': 1, 'word:tag:VERB:sold': 1, 'bi:<t>:NOUN': 1, 'bi:NUM:NOUN': 1, 'word:tag:ADP:in': 1, 'word:tag:NOUN:last': 1, 'tag-1:VERB': 1, 'word:tag:NOUN:year': 1, 'word:year': 1, 'word:tag:NOUN:luxury': 1, 'bi:NOUN:</t>': 1, 'word:The': 1, 'word:1,214': 1, 'word:in': 1, 'word:tag:NOUN:The': 1, 'word:U.S.': 2, 'word:tag:</t>:U.S.': 1, 'word:tag:NOUN:maker': 1, 'tag:</t>': 1, 'word:tag:NOUN:the': 1, 'tag:NOUN': 9, 'tag-1:<t>': 1, 'bi:NOUN:VERB': 1, 'tag-1:NUM': 1, 'tag:VERB': 1, 'bi:VERB:NUM': 1, 'word:maker': 1, 'bi:NOUN:ADP': 1, 'tag-1:NOUN': 9, 'bi:NOUN:NOUN': 6, 'word:last': 1, 'tag:ADP': 1, 'word:tag:NOUN:cars': 1, 'tag:NUM': 1, 'word:tag:NOUN:U.S.': 1, 'word:tag:NOUN:auto': 1, 'tag-1:ADP': 1, 'word:auto': 1, 'word:tag:NUM:1,214': 1, 'word:cars': 1, 'word:the': 1, 'bi:ADP:NOUN': 1}
    avg loss: 0.684211 w: [[  4.   2.   0.   0.   3.   6.   5.  -1.  -1.   2.  -6.   3.   2.   1.
        2.  -7.   0.  -1.   2.   1.   2.   2.   1.  -7.   0.  -1.   2.   1.
        2.   2.   1.  -5.   0.  10.   0.   0.   0.   0.  -5.   0.   0.   0.
        0.   0.   0.   0.   0.   5.  10.   4.   4.   5.   5.   1.   3.   2.
        2.   3.   2.   1.   1.   1.   3.   4.   3.   0.   0.]]
    effective learning rate: 1.000000


.. code:: python

    words = "Ms. Haag plays Elianti".split()
    sp.predict([words])

.. parsed-literal::

    Weights: [{'word:END': 10.0, 'word:tag:VERB:sold': 3.0, 'word:tag:NOUN:Ms.': 2.0, 'word:tag:</t>:END': 10.0, 'word:tag:NOUN:Haag': 3.0, 'bi:DET:NOUN': 6.0, 'bi:NUM:NOUN': 2.0, 'word:tag:ADP:in': 4.0, 'tag-1:ADJ': -1.0, 'word:tag:.:.': 5.0, 'word:tag:NOUN:year': 1.0, 'word:tag:NOUN:luxury': 1.0, 'bi:NOUN:</t>': -1.0, 'word:U.S.': -5.0, 'tag-1:DET': 1.0, 'word:tag:NOUN:Elianti': 1.0, 'bi:.:</t>': 4.0, 'tag:.': -7.0, 'bi:ADP:DET': 3.0, 'word:tag:NOUN:maker': 1.0, 'word:tag:NOUN:auto': 3.0, 'tag:NOUN': 2.0, 'bi:NOUN:VERB': 3.0, 'tag-1:NUM': 2.0, 'word:tag:NUM:1,214': 3.0, 'tag:VERB': 1.0, 'tag:ADJ': -1.0, 'word:tag:VERB:plays': 4.0, 'bi:<t>:DET': 2.0, 'bi:VERB:NUM': 2.0, 'bi:NOUN:ADP': 2.0, 'tag-1:NOUN': 2.0, 'bi:NOUN:NOUN': -6.0, 'bi:NOUN:ADJ': -1.0, 'word:tag:DET:the': 5.0, 'bi:VERB:NOUN': 1.0, 'tag:ADP': 2.0, 'word:tag:NOUN:cars': 2.0, 'tag:NUM': 2.0, 'word:.': -5.0, 'word:tag:NOUN:U.S.': 2.0, 'bi:NOUN:.': 5.0, 'tag-1:.': -7.0, 'tag:DET': 1.0, 'tag-1:ADP': 2.0, 'word:tag:DET:The': 5.0, 'word:tag:ADJ:last': 4.0, 'tag-1:VERB': 1.0}]
    SCORE IS: 31.0
    <t> -> ADP
    ADP -> DET
    DET -> NOUN
    NOUN -> VERB
    VERB -> </t>
    {'word:tag:NOUN:Haag': 1, 'word:tag:</t>:Elianti': 1, 'word:Haag': 1, 'word:Ms.': 1, 'tag-1:VERB': 1, 'bi:<t>:ADP': 1, 'word:Elianti': 2, 'bi:DET:NOUN': 1, 'bi:ADP:DET': 1, 'tag:</t>': 1, 'tag:NOUN': 1, 'tag-1:<t>': 1, 'bi:NOUN:VERB': 1, 'tag:VERB': 1, 'word:tag:VERB:plays': 1, 'word:plays': 1, 'tag-1:NOUN': 1, 'tag:ADP': 1, 'bi:VERB:</t>': 1, 'tag:DET': 1, 'tag-1:DET': 1, 'tag-1:ADP': 1, 'word:tag:DET:Ms.': 1, 'word:tag:ADP:Elianti': 1}




.. parsed-literal::

    [{Bigram(position=-1, prevtag='<t>', tag='ADP'),
      Bigram(position=0, prevtag='ADP', tag='DET'),
      Bigram(position=1, prevtag='DET', tag='NOUN'),
      Bigram(position=2, prevtag='NOUN', tag='VERB'),
      Bigram(position=3, prevtag='VERB', tag='</t>')}]



.. code:: python

    c = Counter()
    c["ell"] += 20
    c.keys()



.. parsed-literal::

    ['ell']



.. code:: python

    # from  pystruct.plot_learning import plot_learning
    # plot_learning(sp)