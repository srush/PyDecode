"""
Classes for sequence tagging/labeling problem.
"""
import pydecode
import pydecode.nlp.decoding as decoding
import itertools
import numpy as np

class TaggingProblem(decoding.DecodingProblem):
    """
    Description of a tagging program.

    Given an input sequence x_0,x_1, ... x_{n-1}, predict the best
    output sequence y_0,y_1, ... y_{n-1}.
    """
    def __init__(self, size, tag_sizes=None, pruned_tag_sizes=None):
        r"""
        Describe a tagging problem.

        Parameters
        ----------
        size : int
           The length n of the underlying sequence.

        tag_sizes : sequence of ints
           The number of tags available at each position.
           y_0 \in tag_sizes[0] etc.

        """
        self.size = size
        if tag_sizes is not None:
            self.tag_sizes = tag_sizes
            self.max_tag_size = max(tag_sizes)
            assert tag_sizes[0] == 1
            assert tag_sizes[size-1] == 1
        else:
            self.pruned_tag_sizes = pruned_tag_sizes
            self.max_tag_size = max([max(size) for size in pruned_tag_sizes]) + 1
            assert len(pruned_tag_sizes[0]) == 1, pruned_tag_sizes[0]
            assert len(pruned_tag_sizes[self.size-1]) == 1

    def feasible_set(self):
        """
        Generate all valid tag sequences for a tagging problem.
        """
        for seq in itertools.product(*map(range, self.tag_sizes)):
            yield seq



def ngrams(tag_sequence, K):
    """
    Generates all n-grams from a sequence.

    Parameters
    ----------
    K : int
       The size of the n-gram.

    Yields
    ------
    contexts : tuple (i, t_i, t_{i-1}, t_{i - (K-1)})
    """
    for i, t in enumerate(tag_sequence):
        yield (i, t) + tuple([(tag_sequence[i-k] if i - k >= 0 else "*START*")
                               for k in range(K-1, 0, -1)])

class BiTagCoder(object):
    """
    Codes a tag sequence as a sparse output array as a 3 x n
    array of sparse elements of the form (i, t_i, t_{i-1}).
    """
    POS = 0
    TAG = 1
    PREV_TAG = 2

    START = 0
    END = 1

    def __init__(self):
        self.tag_encoder = None
        self.tags = None

    def shape(self, problem):
        return [problem.size,
                problem.max_tag_size,
                problem.max_tag_size]

    def pretty(self, output):
        return output[0], self._tag_trans(output[self.PREV_TAG], 1), self._tag_trans(output[self.TAG], 1)

    def fit(self, Y):
        self.tag_encoder = {}
        self.rev_tag_encoder = {}
        tags = set()
        for y in Y:
            tags.update(y)
        tags.remove("*START*")
        # tags.remove("*END*")
        self.tag_encoder["*START*"] = 0
        self.tag_encoder["*END*"] = 1
        self.rev_tag_encoder[0] = "*START*"
        self.rev_tag_encoder[1] = "*END*"
        for i, t in enumerate(tags, 2):
            self.tag_encoder[t] = i
            self.rev_tag_encoder[i] = t
        self.tags = len(self.tag_encoder)

    def _tag_trans(self, tag, d=0):
        if self.tag_encoder is not None:
            # HACKY. FIX THIS
            # if tag == "*START*"  and d == 0: return 0
            # if tag == "*END*"  and d == 0: return 1
            # if tag == 0 and d == 1: return "*START*"
            # if tag == 1 and d == 1: return "*END*"
            return self.tag_encoder[tag] if d == 0 else \
                self.rev_tag_encoder[tag]
        else:
            return tag


    def transform(self, sequence):
        """
        Maps a tag sequence to output array.
        """
        seq = [self._tag_trans(t) for t in ("*START*",) + sequence]
        return np.array([[i, con[0], con[1]]
                         for i, con in enumerate(zip(seq[1:], seq))])

    def inverse_transform(self, outputs):
        """
        Maps an output array to a tag sequence.
        """
        sequence = [None] * (len(outputs) + 1)
        sequence[0] = self._tag_trans(self.START, 1)
        for output in outputs:
            sequence[output[self.POS]] = \
                self._tag_trans(output[self.TAG], 1)
        return tuple(sequence)

class BigramTagger(decoding.HypergraphDecoder):
    """
    Bigram tagging decoder.
    """
    def output_coder(self):
        """
        Get the output coder for a tagging problem.

        Parameters
        ----------
        problem : TaggingProblem

        """
        return BiTagCoder()

    def dynamic_program(self, problem):
        """
        Construct a dynamic program for decoding a tagging problem.

        Parameters
        ----------
        problem : TaggingProblem

        Returns
        --------
        dp : DynamicProgram

        """

        n = problem.size
        K = problem.tag_sizes


        t = problem.max_tag_size
        coder = np.arange(n * t, dtype=np.int64)\
            .reshape([n, t])
        out = np.arange(n * t * t, dtype=np.int64)\
            .reshape([n, t, t])

        c = pydecode.ChartBuilder(coder, out,
                                  unstrict=True,
                                  lattice=True)

        c.init(coder[0, :K[0]])
        for i in range(1, problem.size):
            for t in range(K[i]):
                c.set(coder[i, t],
                      coder[i-1, :K[i-1]],
                      out=out[i, t, :K[i-1]])

        return c.finish(False)


class PrunedBigramTagger(decoding.HypergraphDecoder):
    """
    Bigram tagging decoder.
    """
    def output_coder(self):
        return BiTagCoder()

    def __init__(self, expected_tags=None):
        self._use_cache = (expected_tags is not None)
        self._expected_tags = expected_tags
        if self._use_cache:
            self._cache_outputs = {}
            self._cache_coder = {}

            t = expected_tags
            for n in range(50):
                self._cache_outputs[n] = np.arange(n * t * t, dtype=np.int64)\
                    .reshape([n, t, t])
                self._cache_coder[n] =  np.arange(n * t, dtype=np.int64) \
                    .reshape([n, t])

    def dynamic_program(self, problem):
        """
        Construct a dynamic program for decoding a tagging problem.

        Parameters
        ----------
        problem : TaggingProblem

        Returns
        --------
        dp : DynamicProgram

        """

        n = problem.size
        K = problem.pruned_tag_sizes
        t = problem.max_tag_size \
            if not self._use_cache \
            else self._expected_tags

        if self._use_cache and n in self._cache_coder:
            coder = self._cache_coder[n]
            out = self._cache_outputs[n]
        else:
            coder = np.arange(n * t, dtype=np.int64)\
                .reshape([n, t])
            out = np.arange(n * t * t, dtype=np.int64)\
                .reshape([n, t, t])
            if self._use_cache:
                self._cache_coder[n] = coder
                self._cache_outputs[n] = out

        c = pydecode.ChartBuilder(coder, out,
                                  unstrict=True,
                                  lattice=True)

        c.init(coder[0, [0]])
        for i in range(1, problem.size):
            for t in K[i]:
                c.set(coder[i, t],
                      coder[i-1, K[i-1]],
                      out=out[i, t, K[i-1]])

        return c.finish(False)

class PrunedBigramTagger2(decoding.HypergraphDecoder):
    """
    Bigram tagging decoder.
    """
    def output_coder(self):
        return BiTagCoder()

    def dynamic_program(self, problem):
        """
        Construct a dynamic program for decoding a tagging problem.

        Parameters
        ----------
        problem : TaggingProblem

        Returns
        --------
        dp : DynamicProgram

        """

        n = problem.size
        K = problem.pruned_tag_sizes

        t = problem.max_tag_size
        coder = np.arange(n * t, dtype=np.int64)\
            .reshape([n, t])
        out = np.arange(n * t * t, dtype=np.int64)\
            .reshape([n, t, t])

        for i in range(problem.size):
            for t in range(len(K[i])):
                coder[i, t] = (i, K[i][t])

        c = pydecode.ChartBuilder(coder, out,
                                  unstrict=True,
                                  lattice=True)

        c.init(coder[0, [0]])
        for i in range(1, problem.size):
            for t in K[i]:
                c.set(coder[i, t],
                      coder[i-1, :K[i-1]],
                      out=out[i, t, :K[i-1]])

        return c.finish(False)
