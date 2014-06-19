import pydecode as ph
import pydecode.nlp.decoding as decoding
from collections import namedtuple
import itertools
import random

START = "<s>"

class TaggingProblem(decoding.DecodingProblem):
    def __init__(self, size, order, tag_set):
        self.size = size
        self.order = order
        self.tag_set = tag_set

    def feasible_set(self):
        for seq in itertools.product(self.tag_set, repeat=self.size):
            yield TagSequence(seq)

class TagSequence(object):
    def __init__(self, tags):
        self.tags = tuple(tags)

    def __eq__(self, other):
        return self.tags == other.tags

    def __cmp__(self, other):
        return cmp(self.tags, other.tags)

    def __repr__(self):
        return str(self.tags)

    def contexts(self, back):
        for i, t in enumerate(self.tags):
            yield (i, t) + (tuple([(self.tags[i-k] if i - k >= 0 else START)
                                   for k in range(back, 0, -1)]),)

class TagScorer(decoding.Scorer):
    def __init__(self, problem, scores):
        self._scores = scores
        self._problem = problem

    def tag_score(self, i, tag, context):
        return self._scores[i][tag][context]

    def score(self, instance):
        return sum((self.tag_score(*con)
                    for con in instance.contexts(self._problem.order-1)))

    @staticmethod
    def random(problem):
        return TagScorer(problem,
                         [[{t2 : random.random()
                            for t2 in itertools.product(problem.tag_set + [START],
                                                        repeat=problem.order-1)}
                           for t in problem.tag_set]
                          for i in range(problem.size)])


class Tagged(namedtuple("Tagged", ["position", "tag", "memory"])):
    pass

class NGram(namedtuple("NGram", ["position", "tag", "memory"])):
    def __str__(self): return "%s %s"%(self.word, self.tag)

class Tagger(decoding.HypergraphDecoder):
    def path_to_instance(self, problem, path):
        sequence = [None] * problem.size
        for node in path.nodes:
            print node.label
            if isinstance(node.label, Tagged):
                sequence[node.label.position] = node.label.tag
        return TagSequence(sequence)

    def potentials(self, hypergraph, scorer):
        def score(edge):
            label = edge.label
            if label is None: return 0.0
            return scorer.tag_score(label.position, label.tag, label.memory)
        return ph.LogViterbiPotentials(hypergraph).from_vector([
                score(edge) for edge in hypergraph.edges])


class BigramTagger(Tagger):
    def dynamic_program(self, c, problem):
        assert(problem.order == 2)
        c.init("START")
        for t in problem.tag_set:
            c[Tagged(0, t, ())] = \
                c.sum([c["START"] * c.sr(NGram(0, t, (START,)))])

        for i in range(1, problem.size):
            for t in problem.tag_set:
                c[Tagged(i, t, ())] = \
                    c.sum([c[key] * c.sr(NGram(i, t, (t2,)))
                           for t2 in problem.tag_set
                           for key in [Tagged(i-1, t2, ())]
                           if key in c])
        c["END"] = c.sum([c[Tagged(problem.size - 1, t, ())]
                          for t in problem.tag_set])


class TrigramTagger(Tagger):
    def dynamic_program(self, c, problem):
        assert(problem.order == 3)
        c.init("START")
        for t in problem.tag_set:
            c[Tagged(0, t, (START,))] = \
                c.sum([c["START"] * c.sr(NGram(0, t, (START, START,)))])

        for i in range(1, problem.size):
            for t in problem.tag_set:
                for t2 in problem.tag_set:
                    c[Tagged(i, t, (t2,))] = \
                        c.sum([c[key] * c.sr(NGram(i, t, (t3, t2,)))
                               for t3 in problem.tag_set + [START]
                               for key in [Tagged(i-1, t2, (t3,))]
                               if key in c])
        c["END"] = c.sum([c[Tagged(problem.size - 1, t, (t2,))]
                          for t in problem.tag_set
                          for t2 in problem.tag_set])


        # words = ["ROOT"] + sentence.strip().split(" ") + ["END"]
        # c.init(Tagged(0, "ROOT", "ROOT"))
        # for i, word in enumerate(words[1:], 1):
        #     prev_tags = emission[words[i-1]].keys()
        #     for tag in emission[word].iterkeys():
        #         c[Tagged(i, word, tag)] = \
        #             c.sum([c[key] * c.sr(Bigram(word, tag, prev))
        #                    for prev in prev_tags
        #                    for key in [Tagged(i - 1, words[i - 1], prev)]
        #                    if key in c])


# def tag_first_order(n, tags):
#     lattice = ph.make_lattice(n, len(tags))
#     lattice.labeling = ph.Labeling(lattice, [Bigram(node.label.i, node.label.j)
#                                              for node in lattice nodes])
#     return lattice

# def tag_second_order(n, tags):
#     lattice = ph.make_lattice(n, len(tags))
#     lattice.labeling = ph.Labeling(lattice, [Bigram(node.label.i, node.label.j)
#                                              for node in lattice nodes])
#     return lattice
