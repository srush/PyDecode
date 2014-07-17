from collections import Counter, defaultdict
import re
import pandas as pd
from StringIO import StringIO
import numpy as np

def read_csv_records(f, front=[], back =[]):
    s = open(f).read()
    for l in re.finditer("(.*?)\n\n", s, re.DOTALL):
        yield np.array(front + [line.split()
                        for line in l.group(1).split("\n")] + back)

CONLL = {"INDEX":1,
         "WORD":1,
         "TAG":3,
         "HEAD":6,
         "HEAD":7}

TAG = {"WORD":0,
       "TAG":1}


# def pad(records, front=[], back =[]):
#     for record in records:
#         record = np.append(record, back)
#         record = np.insert(record, 0, front)

# class Sentence:
#     def __init__(self, words, properties):
#         self.words = words
#         self.properties = properties

#     def word(self, index):
#         return self.words[index]

#     def padded_words(self):
#         return ["*START*"] + self.words + ["*END*"]

#     def padded_tags(self):
#         return ["*START*"] + self.properties["tags"] + ["*END*"]

#     def padded_tags(self):
#         return ["*START*"] + self.properties["tags"] + ["*END*"]

#     def rooted_heads(self):
#         return [-1] + self.properties["heads"]


# # class Lexicon:
# #     def __init__(self):
# #         pass

# #     def initialize(self, counts, word_counts, tag_set):
# #         self.counts = counts
# #         self.word_counts = word_counts
# #         self.tag_set = tag_set
# #         self.tag_num = {tag: i for i, tag in enumerate(tag_set)}
# #         self.word_num = {word: i
# #                          for i, word in enumerate(word_counts.iterkeys())}

# #     @staticmethod
# #     def build_lexicon(corpus):
# #         tag_set = set()
# #         word_counts = Counter()
# #         counts = defaultdict(Counter)
# #         for sentence in corpus:
# #             for word in sentence:
# #                 counts[word.lex][word.tag] += 1
# #                 word_counts[word.lex] += 1
# #                 tag_set.add(word.tag)
# #         return Lexicon().initialize(counts, word_counts, tag_set)


# # class Corpus:
# #     def __init__(self, sentences):
# #         self.sentences = sentences

# #     def __iter__(self):
# #         return iter(self.sentences)


# # class Word:
# #     def __init__(self, lex, ident, tag=None):
# #         self.lex = lex
# #         self.ident = ident
# #         self.tag = tag




# class DependencySentence:
#     def __init__(self, words, dependencies):
#         self.words = words
#         self.dependencies = dependencies

#     def word(self, index):
#         return self.words[index]

#     def head(self, index):
#         return self.dependencies[index]
