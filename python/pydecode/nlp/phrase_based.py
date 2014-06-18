from collections import namedtuple, defaultdict
import itertools


class PhraseTranslation:
    def __init__(self):
        pass


class PhraseScore:
    def __init__(self, phrase_score):
        pass
    def score():
        pass

class Phrase(namedtuple("Phrase", ["source_span", "target_words"])):
    @property
    def last(self):
        return self.target_words[-1]

    @property
    def src_len(self):
        return self.source_span[1] - self.source_span[0]

    def __str__(self):
        return "(%d, %d) %s"%(self.source_span[0], self.source_span[1], 
                              self.target_words)

class State(namedtuple("State", ["num_source", "last"])):
    def __str__(self):
        return "(%d, %d)"%(self.num_source, self.last)

def make_phrase_table(phrases):
    d = defaultdict(list)
    for phrase in phrases:
        d[phrase.last].append(phrase)
    return d

def phrase_lattice(n, phrase_table, words, c):
    c.init(State(0, 0))
    for i in range(1, n):
        for last in range(n):
            c[State(i, last)] = \
                c.sum([c[key] * c.sr(phrase)
                       for phrase in phrase_table[last]
                       for w in words
                       for num in [i - phrase.src_len]
                       if num >= 0
                       for key in [State(num, w)]
                       if key in c])
    c["END"] = \
        c.sum([c[key] 
               for w in words
               for key in [State(n-1, w)]
               if key in c
               ])
