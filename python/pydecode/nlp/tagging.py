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
    def __init__(self, size, tag_sizes):
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
        self.tag_sizes = tag_sizes
        self.max_tag_size = max(tag_sizes)
        assert tag_sizes[0] == 1
        assert tag_sizes[size-1] == 1

    def feasible_set(self):
        """
        Generate all valid tag sequences for a tagging problem.
        """
        for seq in itertools.product(*map(range, self.tag_sizes)):
            yield TagSequence(seq)

class TagSequence(object):
    """
    A tag sequence y_0,y_1...y_{n-1}.
    """
    def __init__(self, tags):
        self.tags = tuple(tags)

    def __eq__(self, other):
        return self.tags == other.tags

    def __cmp__(self, other):
        return cmp(self.tags, other.tags)

    def __repr__(self):
        return str(self.tags)

    def ngrams(self, K):
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
        for i, t in enumerate(self.tags):
            yield (i, t) + tuple([(self.tags[i-k] if i - k >= 0 else 0)
                                   for k in range(K-1, 0, -1)])

class BiTagCoder(object):
    """
    Codes a tag sequence as a sparse output array as a 3 x n
    array of sparse elements if tge form (i, t_i, t_{i-1}).
    """

    def __init__(self, problem):
        self._problem = problem
        self.shape_ = [problem.size,
                       problem.max_tag_size,
                       problem.max_tag_size]

    def transform(self, sequence):
        """
        Maps a TagSequence to output array.
        """
        return np.array([[con[0], con[1], con[2]]
                         for con in sequence.ngrams(2)])

    def inverse_transform(self, tags):
        """
        Maps an output array to a TagSequence.
        """
        sequence = [None] * self._problem.size
        sequence[0] = 0
        for (i, t, pt) in tags:
            sequence[i] = t
        return TagSequence(sequence)

class BigramTagger(decoding.HypergraphDecoder):
    """
    Bigram tagging decoder.
    """
    def output_coder(self, problem):
        """
        Get the output coder for a tagging problem.

        Parameters
        ----------
        problem : TaggingProblem

        """
        return BiTagCoder(problem)

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
