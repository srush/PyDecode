import pydecode.hyper as ph
from pydecode.semiring import *


class ChartBuilder:
    """
    A dynamic programming chart parameterized by semiring.

    """

    def __init__(self, score_fn,
                 semiring=ProbSemiRing,
                 build_hypergraph=False):
        """
        Initialize the dynamic programming chart.

        Parameters
        ------------

        score_fn :  label -> "score"
           A function from edge label to score.

        semiring : :py:class:`SemiRing`
           The semiring class to use.

        build_hypergraph : bool
           Should we build a hypergraph in the chart.

        """
        self._builder = build_hypergraph
        self._chart = {}
        self._semiring = semiring
        self._scorer = score_fn
        self._done = False
        self._last = None
        if self._builder:
            self._hypergraph = ph.Hypergraph()
            self._build = self._hypergraph.builder()
            self._build.__enter__()

    def finish(self):
        """
        Finish the chart and get out the root value.
        """

        if self._done:
            raise Exception("Hypergraph not constructed")
        if self._builder:
            self._done = True
            self._build.__exit__(None, None, None)
            return self._hypergraph
        else:
            return self._chart[self._last].unpack()

    def sr(self, label):
        """

        Parameters
        ------------

        label : edge label
           Get the semiring value of the label.
        """
        return self._semiring.make(self._scorer(label))

    def init(self, node_label):
        """
        Initialize a chart cell to 1.

        Parameters
        ------------

        """

        if self._builder:
            node = self._build.add_node([], label=node_label)
            self._chart[node_label] = HypergraphSemiRing([], [node], None)
        else:
            self._chart[node_label] = self._semiring.one()
        return self._chart[node_label]

    def sum(self, edges):
        "Combine values with + semiring operation."
        return sum(edges, self._semiring.zero())

    def __setitem__(self, label, val):
        if label in self._chart:
            raise Exception(
                "Chart already has label {}".format(label))
        if self._builder:
            if not val.is_zero():
                node = self._build.add_node(val.edge_list,
                                            label=label)
                self._chart[label] = \
                    HypergraphSemiRing([], [node], None)
        else:
            if not val.is_zero():
                self._chart[label] = val
        self._last = label
        return self._chart[label]

    def __contains__(self, label):
        return label in self._chart

    def __getitem__(self, label):
        return self._chart.get(label, self._semiring.zero())
