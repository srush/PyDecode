
import pydecode
import random

# class SizedTupleHasher:
#     """
#     For hashing chart items to integers.
#     """
#     def __init__(self, sizes):
#         self._sizes = sizes
#         multi = 1
#         self._multipliers = []
#         for s in sizes:
#             self._multipliers.append(multi)
#             multi *= s

#         self._max_size = multi

#     def __call__(self, tupl):
#         val = 0
#         multiplier = 1
#         for i in range(len(tupl)):
#             val += self._multipliers[i] * tupl[i]
#         return val

#     def max_size(self):
#         return self._max_size

#     def unhash(self, val):
#         tupl = ()
#         v = val
#         for i in range(len(self._multipliers)-1, -1, -1):
#             tupl = (v / self._multipliers[i],) + tupl
#             v = v % self._multipliers[i]
#         return tupl


# def test_sized_hasher():
#     a = random.randint(1, 40)
#     b = random.randint(1, 40)
#     c = random.randint(1, 40)
#     hasher = ph.SizedTupleHasher([a,b,c])
#     #hasher = SizedTupleHasher([a,b,c])
#     d = set()
#     for i in range(a):
#         for j in range(b):
#             for k in range(c):
#                 v = hasher.hasher((i,j,k))
#                 print (i,j,k), v
#                 assert(v not in d)
#                 d.add(v)
