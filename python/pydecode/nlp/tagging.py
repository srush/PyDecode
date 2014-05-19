import pydecode.nlp.decoding as decoding

START = "<s>"

class TaggingProblem(decoding.DecodingProblem):
    def __init__(self, size, order, tag_set):
        self.size = size
        self.order = order
        self.tag_set = tag_set

    def feasible_set(self):
        for seq in itertools.product(self.tag_set, repeat=self.size):
            return TagSequence(seq)

class TagSequence(object):
    def __init__(self, tags):
        self.tags = tags

    def __eq__(self, other):
        return self.tags == other.tags

    def __cmp__(self, other):
        return cmp(self.tags, other.tags)

    def __repr__(self):
        return str(self.tags)
            
    def contexts(self, back):
        for i, t in enumerate(self.tags):
            return (i, t) + tuple([(tags[i-k] if i - k >= 0 else START) 
                                   for k in range(back, 0, -1)]) 
       
class TagScorer(decoding.Scorer):
    def __init__(self, problem, scores):
        self._scores = scores

    def tag_score(self, i, tag, context):
        return self._scores[i][tag][context]

    def score(self, instance):
        return sum((self.tag_score(*con) 
                    for con in instance.contexts()))

    @staticmethod
    def random(problem):
        return TagScorer(problem, 
                         [[{t2 : random.random()
                            for t2 in itertools.product(problem.tag_set, 
                                                        repeat=problem.order-1)}
                           for t in tagset]
                          for i in range(problem.size)])


class Tagged(namedtuple("Tagged", ["position", "tag", "memory"])):
    def __str__(self): return "%s %s"%(self.word, self.tag)

class Tagger(decoding.HypergraphDecoder):
    def path_to_instance(self, problem, path):        
        sequence = [None] * problem.size
        for node in path.nodes:
            if node.label is not None:
                sequence[node.label.position] = node.label.tag
        return TagSequence(sequence)

    def potentials(self, hypergraph):
        def score(edge):
            label = edge.head.label
            if label is None: return 0.0
            return scorer.arc_score(label.position, label.tag, label.memory)
        return LogViterbiPotentials(hypergraph).from_vector([
                score(edge) for edge in hypergraph.edges])
        

class BigramTagger(Tagger):
    def dynamic_program(self, c):
        words = ["ROOT"] + sentence.strip().split(" ") + ["END"]
        c.init(Tagged(0, "ROOT", "ROOT"))    
        for i, word in enumerate(words[1:], 1):
            prev_tags = emission[words[i-1]].keys()
            for tag in emission[word].iterkeys():
                c[Tagged(i, word, tag)] = \
                    c.sum([c[key] * c.sr(Bigram(word, tag, prev)) 
                           for prev in prev_tags 
                           for key in [Tagged(i - 1, words[i - 1], prev)] 
                           if key in c])
        return c

class TrigramTagger(Tagger):
    def dynamic_program(self, c):
        words = ["ROOT"] + sentence.strip().split(" ") + ["END"]
        c.init(Tagged(0, "ROOT", "ROOT"))    
        for i, word in enumerate(words[1:], 1):
            prev_tags = emission[words[i-1]].keys()
            for tag in emission[word].iterkeys():
                c[Tagged(i, word, tag)] = \
                    c.sum([c[key] * c.sr(Bigram(word, tag, prev)) 
                           for prev in prev_tags 
                           for key in [Tagged(i - 1, words[i - 1], prev)] 
                           if key in c])


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
