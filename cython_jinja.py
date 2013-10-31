from jinja2 import Environment, PackageLoader

vars = {"semirings":
 [{"type": "Viterbi", "ptype": "ViterbiW",
   "ctype": "ViterbiWeight", "vtype": "double",
   "float" : True},
  {"type": "LogViterbi", "ptype": "LogViterbiW",
   "ctype": "LogViterbiWeight", "vtype": "double",
   "float" : True}]
}

env = Environment(loader=PackageLoader('pydecode', 'templates'))
template = env.get_template('hyper.pyx.tpl')
out = open("python/pydecode/hyper.pyx", "w")
print >>out, template.render(vars)
