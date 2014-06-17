
## Tutorial 5: Training a CRF

# In[1]:

from sklearn.feature_extraction import DictVectorizer
from collections import namedtuple
import pydecode.model as model
import pydecode.chart as chart
import pydecode.hyper as ph
from collections import Counter, defaultdict
from itertools import izip
import warnings


# In[2]:

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

    


# In[3]:

class Bigram(namedtuple("Bigram", ["position", "prevtag", "tag"])):
    def __str__(self): return "%s -> %s"%(self.prevtag, self.tag)
    
    @staticmethod
    def from_tagging(tagging):
        return [Bigram(i, tag=tag, prevtag=tagging[i-1] if i > 0 else "<t>")
                for i, tag in enumerate(tagging)] + [Bigram(len(tagging), tag="</t>", prevtag=tagging[-1])] 
      
class Tagged(namedtuple("Tagged", ["position",  "tag"])):
    def __str__(self): return "%s"%(self.tag,)


# In[4]:

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
                c[Tagged(i, tag)] =                     c.sum([c[key] * c.sr(Bigram(i - 1, prev, tag))
                           for prev in prev_tags 
                           for key in [Tagged(i - 1, prev)] 
                           if key in c])
        return c

    def initialize_features(self, sentence):
        return [self.dictionary.word_id(word) for word in sentence]

    def factored_joint_feature(self, sentence, bigram, data):
        word = sentence[bigram.position] if bigram.position < len(sentence) else "END"
        return {"word:tag:%s:%s" % (bigram.tag, word) : 1, 
                "suff:word:tag:%d:%s:%s" % (1, bigram.tag, word[-1:]) : 1, 
                "suff:word:tag:%d:%s:%s" % (2, bigram.tag, word[-2:]) : 1, 
                "suff:word:tag:%d:%s:%s" % (3, bigram.tag, word[-3:]) : 1, 
                "pre:word:tag:%d:%s:%s" % (1, bigram.tag, word[:1]) : 1, 
                "pre:word:tag:%d:%s:%s" % (2, bigram.tag, word[:2]) : 1, 
                "pre:word:tag:%d:%s:%s" % (3, bigram.tag, word[:3]) : 1, 
                "word:%s" %  word : 1, 
                "tag-1:%s" % bigram.prevtag : 1, 
                "tag:%s" % bigram.tag : 1,
                "bi:%s:%s" % (bigram.prevtag, bigram.tag): 1,
                }


# In[5]:

data_X = map(lambda a: a.split(),
             ["the dog walked",
              "in the park",
              "in the dog"])
data_Y = map(lambda a: Bigram.from_tagging(a.split()),
             ["D N V", "I D N", "I D N"])


# In[35]:

# def parse_training(handle):
#     x = []
#     y = []
#     for l in handle:
#         if not l.strip():
#             yield (x, y)
#             x = []
#             y = []
#         else:
#             word, tag = l.split()
#             x.append(word)
#             y.append(tag)
#     yield (x, y)
# data_X, data_Y = zip(*parse_training(open("tag/tag_train_small.dat")))
# data_Y = [Bigram.from_tagging(t) for t in data_Y] 


# In[6]:

hm = TaggingCRFModel()
hm.initialize(data_X, data_Y)
for i in range(len(data_X))[:10]:
    s = set(data_Y[i])
    c = chart.ChartBuilder(lambda a: a,
                           chart.HypergraphSemiRing, True)
    hm.dynamic_program(data_X[i], c)
    h = c.finish()
    bool_pot = ph.BoolPotentials(h).from_vector(edge.label in s for edge in h.edges)
    path = ph.best_path(h, bool_pot)
    #for edge in path: print h.label(edge)
    assert bool_pot.dot(path)


# Out[6]:

#     set(['I', 'V', 'D', 'N'])
# 

# In[7]:

print data_Y[0]


# Out[7]:

#     [Bigram(position=0, prevtag='<t>', tag='D'), Bigram(position=1, prevtag='D', tag='N'), Bigram(position=2, prevtag='N', tag='V'), Bigram(position=3, prevtag='V', tag='</t>')]
# 

# In[8]:

from pystruct.learners import StructuredPerceptron
hm = TaggingCRFModel()
sp = StructuredPerceptron(hm, verbose=1, max_iter=25)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    sp.fit(data_X, data_Y)


# Out[8]:

#     set(['I', 'V', 'D', 'N'])
#     iteration 0
#     avg loss: 0.666667 w: [ 0.  0.  2.  1.  1.  1.  0.  0.  1.  1.  1.  1.  1.  0.  1.  1.  1.  1.
#       1.  0.  1.  1.  1.  1.  1.  0.  1.  1.  1.  1.  1.  0.  1.  1.  1.  1.
#       1.  0.  1.  1.  1.  1.  1.  0.  0. -2.  2.  0.  0.  0. -2.  2.  0.  0.
#       0.  0.  0.  0.  1.  1.  1.  1.  1.  0.  0.]
#     effective learning rate: 1.000000
#     iteration 1
#     avg loss: 0.000000 w: [ 0.  0.  2.  1.  1.  1.  0.  0.  1.  1.  1.  1.  1.  0.  1.  1.  1.  1.
#       1.  0.  1.  1.  1.  1.  1.  0.  1.  1.  1.  1.  1.  0.  1.  1.  1.  1.
#       1.  0.  1.  1.  1.  1.  1.  0.  0. -2.  2.  0.  0.  0. -2.  2.  0.  0.
#       0.  0.  0.  0.  1.  1.  1.  1.  1.  0.  0.]
#     effective learning rate: 1.000000
#     Loss zero. Stopping.
# 

# In[9]:

words = "Ms. Haag plays Elianti .".split()
sp.predict([words])


# Out[9]:

#     [{Bigram(position=0, prevtag='<t>', tag='N'),
#       Bigram(position=1, prevtag='N', tag='N'),
#       Bigram(position=2, prevtag='N', tag='N'),
#       Bigram(position=3, prevtag='N', tag='N'),
#       Bigram(position=4, prevtag='N', tag='N'),
#       Bigram(position=5, prevtag='N', tag='</t>')}]

# In[10]:

# c = Counter()
# c["ell"] += 20
# c.keys()


# In[11]:

# from  pystruct.plot_learning import plot_learning
# plot_learning(sp)

