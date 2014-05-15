from collections import Counter, defaultdict


class Lexicon:
    def __init__(self):
        pass

    def initialize(self, counts, word_counts, tag_set):
        self.counts = counts
        self.word_counts = word_counts
        self.tag_set = tag_set
        self.tag_num = {tag: i for i, tag in enumerate(tag_set)}
        self.word_num = {word: i
                         for i, word in enumerate(word_counts.iterkeys())}

    @staticmethod
    def build_lexicon(corpus):
        tag_set = set()
        word_counts = Counter()
        counts = defaultdict(Counter)
        for sentence in corpus:
            for word in sentence:
                counts[word.lex][word.tag] += 1
                word_counts[word.lex] += 1
                tag_set.add(word.tag)
        return Lexicon().initialize(counts, word_counts, tag_set)


class Corpus:
    def __init__(self, sentences):
        self.sentences = sentences

    def __iter__(self):
        return iter(self.sentences)


class Word:
    def __init__(self, lex, ident, tag=None):
        self.lex = lex
        self.ident = ident
        self.tag = tag


class Sentence:
    def __init__(self, words):
        self.words = words

    def __iter__(self):
        return iter(self.words)

    def word(self, index):
        return self.words[index]


class DependencySentence:
    def __init__(self, words, dependencies):
        self.words = words
        self.dependencies = dependencies

    def word(self, index):
        return self.words[index]

    def head(self, index):
        return self.dependencies[index]
