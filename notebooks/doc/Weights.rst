
.. _weight_types:

Weight Types
============

Each of these algorithms is parameterized over several
different semirings. The ``weight_type`` argument is used to specify
the semiring.

==============  ==============  ===============  ===============  ===============  =======
Name            |splus|           |stimes|       |szero|           |sone|          |stype|
==============  ==============  ===============  ===============  ===============  =======
**LogViterbi**   :math:`\max`    :math:`+`       |ninf|           0                float32
**Viterbi**      :math:`\max`    :math:`*`       0                1                float32
**Real**         :math:`+`       :math:`*`       0                1                float32
**Log**          logsum          :math:`+`       |ninf|           0                float32
**Boolean**      or               and             false           true             uint8
**Counting**     :math:`+`       :math:`*`        0               1                int32
**MinMax**       :math:`\min`    :math:`\max`    |ninf|           |inf|            float32
==============  ==============  ===============  ===============  ===============  =======

.. |stype| replace:: :math:`\mathbb{K}`/dtype
.. |inf| replace:: :math:`\infty`
.. |ninf| replace:: :math:`-\infty`
.. |sone| replace:: :math:`\bar{1}`
.. |szero| replace:: :math:`\bar{0}`
.. |splus| replace:: :math:`\oplus`
.. |stimes| replace:: :math:`\otimes`

Bibliography
------------



.. bibliography:: ../../full.bib 
   :filter: key in {"goodman1998parsing", "allauzen2007openfst"}
   :style: plain

Invariants
----------


Check the semiring properties.

.. code:: python

    import pydecode.test
Check the additive and multiplicative identities.

.. code:: python

    @pydecode.test.property()
    def test_weight_properties(graph, weights, weight_type):
        assert (weight_type.one() * weight_type.from_value(weights[0])).value == weights[0]
        assert (weight_type.zero() + weight_type.from_value(weights[0])).value == weights[0]
    test_weight_properties()