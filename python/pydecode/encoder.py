class OutputEncoder(object):
    """
    Interface for structure encoding. Transforms the natural
    representation of a result into a set of index elements or
    outputs.

    For instance, the natural representation of an n-element POS
    tagging is a sequence of tags from a set T, e.g. `N V N`.  However
    for bigram tagging it is more convenient to encode the output as a
    sequence of indexed-bigrams so that we can assign each a
    score. Each element takes the form (position, cur_tag, prev_tag).

    Formally, we would define the index set I = {1 .. (n-1)} x T^2.
    An output representation consists of a subset of {I}.  For example
    we might encode the above example as [[(1, `V`, N), (2, `N`,
    `V`)].

    For efficiency, we can use the `fit` method to further
    transform the result to a more compact form. In the bigram
    tagging example we might tranform the tags into integers
    i.e. [[(1, 1, 0), (2, 0, 1)].
    """
    def transform(self, y):
        """
        Transform a result into an array of output elements.

        Parameters
        ----------
        y : result

        Returns
        -------
        outputs : array
           Array of outputs.
        """
        raise NotImplementedError()

    def inverse_transform(self, outputs):
        """
        Transform an array of output elements into a result.

        Parameters
        ----------
        outputs : array
           Array of outputs.

        Returns
        -------
        y : result
        """
        raise NotImplementedError()

    def fit(self, Y):
        """
        Fit the encoder to a list of observed results.

        Parameters
        ----------
        Y : list of results
        """
        raise NotImplementedError()
