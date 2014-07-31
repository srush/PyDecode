from jinja2 import Environment, FileSystemLoader
import yaml
import sys

fast_mode = False
if len(sys.argv) > 1 and sys.argv[1] == "fast":
    fast_mode = True

env = Environment(loader=FileSystemLoader('python/pydecode/templates'))

vars = yaml.load(open("python/pydecode/templates/potentials.yaml"))

vars_beam = yaml.load(open("python/pydecode/templates/beam.yaml"))

if fast_mode:
    vars_beam = {"semirings":[]}
    vars["semirings"] = vars["semirings"][:4]
    for var in vars["semirings"][:4]:
        var["ctype"] = var["type"]

else:
    for var in vars["semirings"]:
        var["ctype"] = var["type"]




template = env.get_template('potentials.pyx.tpl')
template_beam = env.get_template('beam.pyx.tpl')
template_chart = env.get_template('chart.pyx.tpl')
out = open("python/pydecode/potentials.pyx", "w")
print >>out, open("python/pydecode/templates/libhypergraph.pyx").read()
# print >>out, open("python/pydecode/templates/extensions.pyx").read()
print >>out, template_chart.render({})
print >>out, template_beam.render(vars_beam)
print >>out, template.render(vars)

template = env.get_template('potentials.pxd.tpl')
template_beam = env.get_template('beam.pxd.tpl')
template_chart = env.get_template('chart.pxd.tpl')
out = open("python/pydecode/potentials.pxd", "w")
print >>out, open("python/pydecode/templates/libhypergraph.pxd").read()
# print >>out, open("python/pydecode/templates/extensions.pxd").read()
print >>out, template_chart.render({})
print >>out, template_beam.render(vars_beam)
print >>out, template.render(vars)
