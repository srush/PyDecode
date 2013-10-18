Hypergraphs are a generalization of lattices for arbitrary dynamic
programming algorithms. In this section we define notation for
hypergraphs and show how they can be used to describe dynamic
programming algorithms and concise linear programs for decdoing
problems. Throughout this work we return to this formalism to easily
move between dynamic programming, hypergraph, and linear programming
representations.

Hypergraph
^^^^^^^^^^

A directed, ordered hypergraph is a pair :math:`({\cal V}, {\cal E})`
where :math:`{\cal V}` is a set of vertices, and :math:`{\cal E}` is a
set of directed hyperedges. Each hyperedge :math:`e \in {\cal E}` is a
tuple :math:`\langle \langle v_2 \ldots v_{|v|} \rangle , v_1 \rangle`
where :math:`v_i \in {\cal V}` for :math:`i \in \{1 \ldots |v|\}`. The
*head* of the hyperedge is :math:`h(e) = v_1`. The *tail* of the
hypergraph is the ordered sequence
:math:`t(e) = \langle v_2 \ldots v_{|v|} \rangle`. The size of the tail
:math:`|t(e)|` may vary across different edges, but
:math:`|t(e)| \geq 1` and :math:`|t(e)| \leq k` for some small constant
:math:`k` for all edges. We represent a directed graph as a directed
hypergraph with :math:`|t(e)| = 1` for all edges :math:`e \in {\cal E}`.

Each vertex :math:`v \in {\cal V}` is either a *non-terminal* or a
*terminal* in the hypergraph. The set of non-terminals is
:math:`{\cal N}=  \{ v \in {\cal V}: h(e) = v \mathrm{\ for\ some\ }  e \in {\cal E}\}`
. Conversely, the set of terminals is defined as
:math:`{\cal T}= {\cal V}\setminus {\cal N}` .

All hypergraphs used in this work are acyclic: informally this implies
that no hyperpath (as defined below) contains the same vertex more than
once (see for a full definition). Acyclicity implies a partial
topological ordering of the vertices. Let this partial order be given by
the inequality operator :math:`\prec`. We also assume there is a
distinguished *root* vertex :math:`1` with the property that
:math:`1\precv` for all :math:`v \in {\cal V}\setminus  \{ 1\}`. For
hyperedges, we use :math:`e \prece'` as shorthand for
:math:`h(e) \prech(e')`.

Define a hyperpath as a tuple
:math:`(x, y) \in  \{0,1\}^{|{\cal V}|} \times  \{0,1\}^{| {\cal E}|}`
where :math:`x(v) =1` if vertex :math:`v` is used in the hyperpath,
:math:`x(v) = 0` otherwise (similarly :math:`y(e) = 1` if hyperedge
:math:`e` is used in the hyperpath, :math:`y(e)= 0` otherwise). A valid
hypergraph satisfies the following constraints:

-  The root vertex must be in the hyperpath, i.e. :math:`x(1) = 1`

-  For every vertex :math:`v \in {\cal N}` visited in the hyperpath,
   :math:`x(v) = 1`, there must be one hyperedge :math:`e` entering the
   vertex, :math:`y(e) = 1` with :math:`h(e) = v`. Conversely, for any
   vertex :math:`v \in
     {\cal N}` not visited, :math:`x(v) = 0`, any edge :math:`e` with
   :math:`h(e) = v` must have :math:`y(e) = 0`. We write this linear
   constraint as

   :math:`x(v) = \sum_{e \in {\cal E}: h(e) = v} y(e)` .

-  For every visited vertex :math:`v \in {\cal V}` other than the root
   with :math:`x(v) = 1`, there must be one leaving hyperedge, i.e.
   :math:`e \in {\cal E}` with :math:`y(e) = 1` and with
   :math:`v \in t(e)` . Conversely, for every non-root vertex :math:`v`
   not visited, :math:`x(v) = 0` , no hyperedge :math:`e\in {\cal E}`
   with :math:`y(e) = 1` can have :math:`v` as one if its children,. We
   write this constraints as ,
   :math:`x(v) = \sum_{e \in {\cal E}: v \in t(e)} y(e)` for all
   :math:`v \in {\cal V}\setminus  \{ 1\}`.

We write the complete path set as

.. math::

   \begin{aligned}
     \mathcal X= \{ (x, y) \in \mathcal X: x(1) &=& 1, \\
     x(v) &=& \sum_{e \in {\cal E}: h(e) = v} y(e) \ \ \ \forall \ v \in {\cal N},  \\
     x(v) &=& \sum_{e \in {\cal E}: v \in t(e)} y(e)\ \ \ \forall \ v \in {\cal V}\setminus  \{ 1\} \}\end{aligned}

The first problem we consider is *unconstrained* hypergraph search. Let
:math:`\theta\in \mathbb{R}^{|{{\cal E}}|}` be the weight vector for the
hypergraph. The unconstrained search problem is to find

.. math:: \max_{(x,y) \in \mathcal X} \sum_{e \in {\cal E}} \theta(e) y(e) = \max_{(x,y) \in \mathcal X} \theta^\top y

This maximization can be computed for any weight vector and directed
acyclic hypergraph in time :math:`O(|{\cal E}|)` (assuming
:math:`|t(e)|` is bounded by a constant for all edges) using simple
bottom-up dynamic programming—essentially the CKY algorithm.
Algorithm [fig:dp] shows this algorithm.

For certain decoding problems it will be convenient to first define a
GLM with derivation set :math:`{\cal X}` and high-level decoding
problem, :math:`\max_{y \in {\cal X}} \theta^{\top}y`. Then when
discussing the details of decoding switch to a hypergraph representation
where we express the same decoding problem as
:math:` \max_{(x,y) \in \mathcal X} \theta^{\top}y` with a one-to-one
mapping between :math:`\mathcal X` and :math:`{\cal X}`.

[h]10cm

:math:`\pi[v_1] \gets s`

[fig:dp]

We can use this algorithm to decode constituency parses, dependency
parses, and syntactic machine translation, as well as a generalization
of the lattices used for speech alignment, part-of-speech tagging, and
head-automata models for dependency parsing.
