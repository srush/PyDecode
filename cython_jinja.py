from jinja2 import Environment, PackageLoader

vars = {"semirings":
 [{"type": "Viterbi", "ptype": "ViterbiW",
   "ctype": "ViterbiWeight", "vtype": "double",
   "float" : True, "viterbi" : True},
  {"type": "LogViterbi", "ptype": "LogViterbiW",
   "ctype": "LogViterbiWeight", "vtype": "double",
   "float" : True, "viterbi" : True},
  {"type": "Inside", "ptype": "InsideW",
   "ctype": "InsideWeight", "vtype": "double",
   "float" : True, "viterbi" : False},
  {"type": "Bool", "ptype": "BoolW",
   "ctype": "BoolWeight", "vtype": "double",
   "bool": True, "viterbi" : True}]
}

env = Environment(loader=PackageLoader('pydecode', 'templates'))
template = env.get_template('hyper.pyx.tpl')
out = open("python/pydecode/hyper.pyx", "w")
print >>out, template.render(vars)
