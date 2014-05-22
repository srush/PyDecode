import pydecode.hyper as ph
import sys


class ChartBuilder:
    """
    A imperative interface for specifying dynamic programs.
    """

    def __init__(self, score_fn=lambda a: a,
                 semiring=None,
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
        self._hasher = lambda a: a
        if self._builder:
            self._hypergraph = ph.Hypergraph()
            self._build = self._hypergraph.builder()
            self._build.__enter__()

        self._list_chart = False

    def set_hasher(self, hasher):
        self._hasher = hasher
        self._chart = [None] * self._hasher.max_size()
        self._list_chart = True

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
        return self._semiring.from_value(self._scorer(label))

    def init(self, label):
        """
        Initialize a chart cell to the 1 value.

        Parameters
        ------------

        label : any
           The node to initialize.
        """
        h = self._hasher(label)
        if self._builder:
            node = self._build.add_node([], label=label)
            self._chart[h] = HypergraphSemiRing(None, [], [node])
        else:
            self._chart[h] = self._semiring.one()
        if self._debug:
            print >>sys.stderr, "Initing", label, label in self
        return self._chart[h]

    def sum(self, edges):
        """
        Combine values with + semiring operation.

        Parameters
        ------------

        edges : any

        """
        return sum(edges, self._semiring.zero())

    def __setitem__(self, label, val):
        h = self._hasher(label)
        if label in self:
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
                self._chart[h] = \
                    HypergraphSemiRing(None, [], [node])
            # else:
            #     self._chart[label] = val.zero()
        else:
            self._chart[h] = val
        self._last = self._hasher(label)
        #return self._chart[label]

    def __contains__(self, label):
        h = self._hasher(label)
        if self._list_chart:
            return self._chart[h] != None
        else:
            return h in self._chart

    def _get(self, h, default):
        if self._list_chart:
            v = self._chart[h]
            return v if v is not None else default
        else:
            return self._chart.get(h, default)

    def __getitem__(self, label):
        h = self._hasher(label)
        if self._strict and label not in self:
            raise Exception("Label not in chart: %s" % (label,))
        if self._debug:
            print >>sys.stderr, "Getting", label, label in self
        return self._get(h, self._semiring.zero())

    def show(self):
        if self._list_chart:
            for i, v in enumerate(self._chart):
                print i, v
        else:
            keys = self._chart.keys()
            keys.sort()
            for key in keys:
                print key, self._chart[key]


INF = 1e8


class HypergraphSemiRing:
    def __init__(self, name=None,
                 edge_list=[], node_list=[], is_zero=False):
        self.edge_list = edge_list
        self.node_list = node_list
        self.name = name
        self._is_zero = is_zero

    @staticmethod
    def from_value(name):
        return HypergraphSemiRing(name)

    def __repr__(self):
        return "%s %s %s %s" % (self.edge_list,
                                self.node_list,
                                self.name,
                                self._is_zero)

    def is_zero(self):
        return self._is_zero

    def __add__(self, other):
        edges = self.edges() + other.edges()
        return HypergraphSemiRing(name=None,
                                  edge_list=edges)

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
