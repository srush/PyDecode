import pydecode.hyper as ph
from pydecode.semiring import *
import sys

class ChartBuilder:
    """
    A dynamic programming chart parameterized by semiring.

    """

    def __init__(self, score_fn=lambda a: a,
                 semiring=ProbSemiRing,
                 build_hypergraph=False,
                 debug=False):
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
        self._debug = debug
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
        Get the semiring value of the label.


        Parameters
        ------------

        label : edge label
           Get the semiring value of the label.
        """
        return self._semiring.make(self._scorer(label))

    def init(self, label):
        """
        Initialize a chart cell to the 1 value.

        Parameters
        ------------

        label : any
           The node to initialize.
        """

        if self._builder:
            node = self._build.add_node([], label=label)
            self._chart[label] = HypergraphSemiRing([], [node], None)
        else:
            self._chart[label] = self._semiring.one()
        if self._debug: 
            print >>sys.stderr, "Initing", label, label in self._chart
        return self._chart[label]

    def sum(self, edges):
        """
        Combine values with + semiring operation.

        Parameters
        ------------

        edges : any

        """
        return sum(edges, self._semiring.zero())

    def __setitem__(self, label, val):
        if label in self._chart:
            raise Exception(
                "Chart already has label {}".format(label))
        if self._builder:
            if not val.is_zero():
                if self._debug:
                    print >>sys.stderr, "Adding node", label
                    for edge in val.edges():
                        print >>sys.stderr, "\t with edge", edge
                node = self._build.add_node(val.edges(),
                                            label=label)
                self._chart[label] = \
                    HypergraphSemiRing([], [node], None)
            else:
                self._chart[label] = val.zero()
        else:
            self._chart[label] = val
        self._last = label
        return self._chart[label]

    def __contains__(self, label):
        return label in self._chart

    def __getitem__(self, label):
        if self._debug: 
            print >>sys.stderr, "Getting",label, label in self._chart
        return self._chart.get(label, self._semiring.zero())
    
    def show(self):
        keys = self._chart.keys()
        keys.sort()
        for key in keys:
            print key, self._chart[key]
