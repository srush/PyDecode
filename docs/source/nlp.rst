==========
NLP
==========

Structured Models
-----------------
=====================  =========  =====================================================
**StructuredEncoder**  |enc|        Encode a structured problem as a hypergraph.
=====================  =========  =====================================================

.. |enc| replace:: [:doc:`doc<notebooks/doc/StructuredEncoder>`]

.. toctree::
   :maxdepth: 2
   :hidden:

   notebooks/doc/StructuredEncoder


Dynamic Programs
----------------

================  =========  =====================================================
**tagger**         |tagger|     Lattice construction for tagging.
**semimarkov**     |semi|      Semi-markov tagging algorithm.
**eisner**         |eisner|     Eisner's algorithm for dependency parsing.
**cfg**            |cfg|       Parsing algorithm for Chomsky normal form grammar.
================  =========  =====================================================

.. |eisner| replace:: [:doc:`doc<notebooks/doc/eisner>`]
.. |tagger| replace:: [:doc:`doc<notebooks/doc/tagger>`]
.. |semi| replace:: [:doc:`doc<notebooks/doc/semimarkov>`]
.. |cfg| replace:: [:doc:`doc<notebooks/doc/cfg>`]

.. toctree::
   :maxdepth: 2
   :hidden:

   notebooks/doc/tagger
   notebooks/doc/semimarkov
   notebooks/doc/eisner
   notebooks/doc/cfg


Training
--------

================  =========  =====================================================
**DPModel**       |dpm|       Lattice construction for tagging.
================  =========  =====================================================

.. |dpm| replace:: [:doc:`doc<notebooks/doc/training>`]
