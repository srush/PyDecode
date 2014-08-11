import numpy as np

class StructuredEncoder(object):
    """
    In structured prediction problems we often
    manipulate abstract output structures such as sequences, graphs,
    and trees. These structures come from a set :math:`{\cal Y}(x)`
    dependent on an input structure :math:`x`.

    In order to work with these structures, we assume that
    :math:`{\cal Y}` is a subset of :math:`\{0,1\}^{\cal I}` where
    :math:`{\cal I}` is an `index set` representing possible ``parts``
    of the output structure.

    For example for a simple unigram POS tagging problem, :math:`I` might be
    :math:`\{1\ldots n\} \times \{1\ldots T\}` where each part :math:`(i, t)`
    is used to indicate whether position i is tagged with tag t.

    In PyDecode, a hypergraph is used to encode all possible output structures,
    each hyperpath represents a particular output structure y, and generally
    each hyperedge label represents a part in :math:`I`.

    A StructuredEncoder is used to map between each of these sets.

    Attributes
    -----------

    shape : tuple of ints
       The size of each dimension of :math:`{\cal I}`.


    """

    def __init__(self, shape):
        self.shape = shape
        self.encoder = np.arange(np.product(shape)).reshape(shape)

    def transform_labels(self, labels):
        """
        Returns the parts represented by the given labels.

        Parameters
        --------

        labels : int ndarray
           The labels to transform. Assumed 0 <= l <= L.

        Returns
        --------

        parts : int ndarray
           The part for each label. The row size is the size of labels.
           The column size is the dimension of parts.
        """
        return np.array(np.unravel_index(labels, self.shape)).T

    def transform_structure(self, structure):
        """
        Returns the parts active in a given structure.

        Parameters
        ----------
        structure : any
            An arbitrary representation of a structure in :math:`{\cal Y}`.

        Returns
        -------
        parts : int ndarray
           The structure represented as a list of active parts.
           The row size is the size of active parts.
           The column size is the dimension of parts.
        """
        raise NotImplementedError()

    def transform_path(self, path):
        """
        Returns the parts in the given path.

        Parameters
        --------

        path : Hyperpath
           The hyperpath.

        Returns
        --------

        y : any
           The corresponding output structure :math:`y`.
        """
        structure = self.transform_labels(path.labeling[path.labeling!=-1])
        return self.from_parts(structure)

    def from_parts(self, parts):
        """
        Returns the output structure corresponding to the active set of parts.

        Parameters
        -------
        parts : int ndarray
           The structure represented as a list of active parts.

        Returns
        --------

        y : any
           The corresponding output structure :math:`y`.

        """
        raise NotImplementedError()

    def random_structure(self):
        raise NotImplementedError()

    def all_structures(self):
        raise NotImplementedError()
