import pydecode.hyper as ph
import sys

class ChartBuilder:
    """
    A imperative interface for specifying dynamic programs.
    """

    def __init__(self, score_fn=lambda a: a,
                 semiring=ph._LogViterbiW,
                 build_hypergraph=False,
                 debug=False,
                 strict=True):
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
        self._strict = strict
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
            return self._chart[self._last].value

    def value(self, label):
        """
        Get the semiring value of the label.


        Parameters
        ------------

        label : edge label
           Get the semiring value of the label.
        """
        return self.sr(label)

    def sr(self, label):
        return self._semiring(self._scorer(label))

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
            self._chart[label] = HypergraphSemiRing(None, [], [node])
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
                    HypergraphSemiRing(None, [], [node])
            else:
                self._chart[label] = val.zero()
        else:
            self._chart[label] = val
        self._last = label
        return self._chart[label]

    def __contains__(self, label):
        return label in self._chart

    def __getitem__(self, label):
        if self._strict and label not in self._chart:
            raise Exception("Label not in chart: %s"%(label,))
        if self._debug:
            print >>sys.stderr, "Getting", label, label in self._chart
        return self._chart.get(label, self._semiring.zero())

    def show(self):
        keys = self._chart.keys()
        keys.sort()
        for key in keys:
            print key, self._chart[key]


"""
THIS CLASS IS DEPRECATED. CODE MOVED TO C++.
"""


INF = 1e8

class SemiRing(object):
    """
    A semiring operation.

    Implements + and *

    """
    def __add__(self, other):
        raise NotImplementedError()

    def __mul__(self, other):
        raise NotImplementedError()

    @classmethod
    def one(cls):
        raise NotImplementedError()

    @classmethod
    def zero(cls):
        raise NotImplementedError()

    @classmethod
    def make(cls, v):
        raise NotImplementedError()


class HypergraphSemiRing(SemiRing):
    def __init__(self, name=None,
                 edge_list=[], node_list=[], is_zero=False):
        self.edge_list = edge_list
        self.node_list = node_list
        self.name = name
        self._is_zero = is_zero

    def __repr__(self):
        return "%s %s %s %s" % (self.edge_list,
                                self.node_list,
                                self.name,
                                self._is_zero)
    def is_zero(self):
        return self._is_zero

    def __add__(self, other):
        return HypergraphSemiRing(name = None,
            edge_list=(self.edges() + other.edges()))

    def edges(self):
        if self.node_list:
            return self.edge_list + [(self.node_list, self.name)]
        else:
            return self.edge_list

    def __mul__(self, other):
        zero = other.is_zero() or self.is_zero()
        if zero:
            return HypergraphSemiRing.zero()
        return HypergraphSemiRing(name=other.name, edge_list=[],
                                  node_list=self.node_list + other.node_list)

    @classmethod
    def one(cls):
        return HypergraphSemiRing()

    @classmethod
    def zero(cls):
        return HypergraphSemiRing(is_zero=True)

    @classmethod
    def make(cls, name):
        return HypergraphSemiRing(name=name)
