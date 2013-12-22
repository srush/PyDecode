from jinja2 import Environment, PackageLoader

vars = {"semirings":
 [{"type": "Viterbi", "ptype": "ViterbiW",
   "ctype": "ViterbiPotential",
   "vtype": "double",
   "intype": "double",
   "viterbi" : True,
   "description": r"""
Weight potentials :math:`R^{{\cal E}}` with :math:`(+, *) = (\max, *)`.

"""
   },
  {"type": "LogViterbi", "ptype": "LogViterbiW",
   "ctype": "LogViterbiPotential",
   "vtype": "double",
   "intype": "double",
   "viterbi" : True,
      "description": r"""
Weight potentials :math:`R^{{\cal E}}` with :math:`(+, *) = (\max, +)`.
"""
},
  {"type": "Inside", "ptype": "InsideW",
   "ctype": "InsidePotential",
   "vtype": "double",
   "intype": "double",
   "viterbi" : True,
   "description": r"""
Weight potentials :math:`R^{{\cal E}}` with :math:`(+, *) = (+, *)`.
"""
},
  {"type": "Bool", "ptype": "BoolW",
   "ctype": "BoolPotential",
   "vtype": "bool",
   "intype": "bool",
   "viterbi" : True,
   "description": r"""
Weight potentials :math:`\{0,1\}^{{\cal E}}` with :math:`(+, *) = (\cap, \cup)`.
"""
},
  {"type": "SparseVector", "ptype": "SparseVectorW",
   "ctype": "SparseVectorPotential",
   "vtype": "vector[pair[int, int]]",
   "intype": "vector[pair[int, int]]",
   "viterbi" : False},
  {"type": "MinSparseVector",
   "ptype": "MinSparseVectorW",
   "ctype": "MinSparseVectorPotential",
   "vtype": "vector[pair[int, int]]",
   "intype": "vector[pair[int, int]]",
   "viterbi" : False},
  {"type": "MaxSparseVector",
   "ptype": "MaxSparseVectorW",
   "ctype": "MaxSparseVectorPotential",
   "vtype": "vector[pair[int, int]]",
   "intype": "vector[pair[int, int]]",
   "viterbi" : False},
  {"type": "BinaryVector", "ptype": "BinaryVectorW",
   "ctype": "BinaryVectorPotential",
   "vtype": "cbitset",
   "intype": "Bitset",
   "viterbi" : False,
   "to_cpp": "val.data",
   "from_cpp": "Bitset().init(val)"
   },

{"type": "Counting", "ptype": "CountingW",
   "ctype": "CountingPotential",
   "vtype": "int",
   "intype": "int",
   "viterbi" : False,
   "description": r"""
"""
   }
#             d = {}
#             cdef vector[pair[int,int]] s= <vector[pair[int,int]]> self.wrap
#             for p in s:
#                 d[p.first] = p.second
#             return d
# """
#    }
  ]
}

env = Environment(loader=PackageLoader('pydecode', 'templates'))

template = env.get_template('potentials.pyx.tpl')
out = open("python/pydecode/potentials.pyx", "w")
print >>out, template.render(vars)

template = env.get_template('potentials.pxd.tpl')
out = open("python/pydecode/potentials.pxd", "w")
print >>out, template.render(vars)
