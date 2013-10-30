
## Tutorial 5: Training a CRF

# In[2]:

from sklearn.feature_extraction import DictVectorizer
from collections import namedtuple
import pydecode.model as model
import pydecode.chart as chart


# In[3]:

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


# In[4]:

class Bigram(namedtuple("Bigram", ["position", "tag", "prevtag"])):
    def __str__(self): return "%s -> %s"%(self.prevtag, self.tag)
    
    @staticmethod
    def from_tagging(tagging):
        return [Bigram(i, tag=tag, prevtag=tagging[i-1])
                for i, tag in enumerate(tagging)]
      
class Tagged(namedtuple("Tagged", ["position", "word", "tag"])):
    def __str__(self): return "%s %s"%(self.word, self.tag)


# In[5]:

def sequence_dynamic_program(sentence, c):
    words = ["ROOT"] + sentence + ["END"]
    c.init(Tagged(0, "ROOT", "ROOT"))
    for i, word in enumerate(words[1:], 1):
        prev_tags = emission[words[i-1]].keys()
        for tag in emission[word].iterkeys():
            c[Tagged(i, word, tag)] =                 c.sum([c[key] * c.sr(Bigram(i - 2, tag, prev))
                       for prev in prev_tags 
                       for key in [Tagged(i - 1, words[i - 1], prev)] 
                       if key in c])
    return c


# In[6]:

c = chart.ChartBuilder(lambda a:a, chart.HypergraphSemiRing, 
                       build_hypergraph = True)
hypergraph = sequence_dynamic_program(["the", "dog"], c).finish()
for edge in hypergraph.edges:
    print hypergraph.label(edge)


# Out[6]:

#     ROOT -> V
#     ROOT -> D
#     ROOT -> N
#     V -> V
#     D -> V
#     N -> V
#     V -> D
#     D -> D
#     N -> D
#     V -> N
#     D -> N
#     N -> N
#     V -> END
#     D -> END
#     N -> END
# 

# In[7]:

class TaggingCRFModel(model.DynamicProgrammingModel):
    def dynamic_program(self, sentence, c):
        return sequence_dynamic_program(sentence, c) 

    def factored_psi(self, sentence, bigram):
        print bigram, sentence
        return {#"word-1:%s"%sentence[bigram.position - 1] if bigram.position != 0 else "", 
                "word:%s" % sentence[bigram.position], 
                "tag-1:%s" % bigram.prevtag, 
                "tag:%s" % bigram.tag}


# In[8]:

data_X = map(lambda a: a.split(),
             ["the dog walked END",
              "in the park END",
              "in the dog END"])
data_Y = map(lambda a: Bigram.from_tagging(a.split()),
             ["D N V", "I D N", "I D N"])

hm = TaggingCRFModel()


# In[13]:

from pystruct.learners import StructuredPerceptron
sp = StructuredPerceptron(hm)
sp.fit(data_X, data_Y)


# Out[13]:


    ---------------------------------------------------------------------------
    ImportError                               Traceback (most recent call last)

    <ipython-input-13-978c39e40e7c> in <module>()
    ----> 1 from pystruct.learners import StructuredPerceptron, SubgradientSSVM
          2 sp = OneSlackSSVM(hm)
          3 sp.fit(data_X, data_Y)


    ImportError: cannot import name SubgradientSSVM


# In[ ]:

#from  pystruct.plot_learning import plot_learning
# plot_learning(sp)


# Out[]:


    ---------------------------------------------------------------------------
    AttributeError                            Traceback (most recent call last)

    <ipython-input-10-43ce9c57b049> in <module>()
          1 from  pystruct.plot_learning import plot_learning
    ----> 2 plot_learning(sp)
    

    /usr/local/lib/python2.7/dist-packages/pystruct/plot_learning.pyc in plot_learning(ssvm, time)
         44     if hasattr(ssvm, 'base_ssvm'):
         45         ssvm = ssvm.base_ssvm
    ---> 46     print("Iterations: %d" % len(ssvm.objective_curve_))
         47     print("Objective: %f" % ssvm.objective_curve_[-1])
         48     inference_run = None


    AttributeError: 'StructuredPerceptron' object has no attribute 'objective_curve_'


#     StructuredPerceptron(average=False, batch=False, decay_exponent=0,
#                decay_t0=10, logger=None, max_iter=100,
#                model=TaggingCRFModel, size_psi: 13, n_jobs=1, verbose=0)
# 
