from jinja2 import Environment, PackageLoader

vars = {"semirings":
 [{"type": "Viterbi", "ptype": "ViterbiW",
   "ctype": "ViterbiPotential", "vtype": "double",
   "float" : True, "viterbi" : True,
   "description": r"""
Weight potentials :math:`R^{{\cal E}}` with :math:`(+, *) = (\max, *)`.

"""
   },
  {"type": "LogViterbi", "ptype": "LogViterbiW",
   "ctype": "LogViterbiPotential", "vtype": "double",
   "float" : True, "viterbi" : True,
      "description": r"""
Weight potentials :math:`R^{{\cal E}}` with :math:`(+, *) = (\max, +)`.
"""
},
  {"type": "Inside", "ptype": "InsideW",
   "ctype": "InsidePotential", "vtype": "double",
   "float" : True, "viterbi" : True,
   "description": r"""
Weight potentials :math:`R^{{\cal E}}` with :math:`(+, *) = (+, *)`.
"""
},
  {"type": "Bool", "ptype": "BoolW",
   "ctype": "BoolPotential", "vtype": "bool",
   "bool": True, "viterbi" : True,
      "description": r"""
Weight potentials :math:`\{0,1\}^{{\cal E}}` with :math:`(+, *) = (\cap, \cup)`.
"""
},
  {"type": "SparseVector", "ptype": "SparseVectorW",
   "ctype": "SparseVectorPotential", "vtype": "vector[pair[int, int]]",
   "bool": False, "viterbi" : False, "float": False,
    "conversion" : "pass"}
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
template = env.get_template('hyper.pyx.tpl')
out = open("python/pydecode/hyper.pyx", "w")
print >>out, template.render(vars)
