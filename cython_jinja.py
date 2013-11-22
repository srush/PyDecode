from jinja2 import Environment, PackageLoader

vars = {"semirings":
 [{"type": "Viterbi", "ptype": "ViterbiW",
   "ctype": "ViterbiPotential", "vtype": "double",
   "float" : True, "viterbi" : True},
  {"type": "LogViterbi", "ptype": "LogViterbiW",
   "ctype": "LogViterbiPotential", "vtype": "double",
   "float" : True, "viterbi" : True},
  {"type": "Inside", "ptype": "InsideW",
   "ctype": "InsidePotential", "vtype": "double",
   "float" : True, "viterbi" : True},
  {"type": "Bool", "ptype": "BoolW",
   "ctype": "BoolPotential", "vtype": "bool",
   "bool": True, "viterbi" : True},
  {"type": "SparseVector", "ptype": "SparseVectorW",
   "ctype": "SparseVectorPotential", "vtype": "vector[pair[int, int]]",
   "bool": False, "viterbi" : False, "float": False}
#    "conversion" : """
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
