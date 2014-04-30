from jinja2 import Environment, PackageLoader
import yaml


env = Environment(loader=PackageLoader('pydecode', 'templates'))

vars = yaml.load(open("python/pydecode/templates/potentials.yaml"))

for var in vars["semirings"]:
    var["ctype"] = var["type"] + "Potential"

vars_beam = yaml.load(open("python/pydecode/templates/beam.yaml"))


template = env.get_template('potentials.pyx.tpl')
template_beam = env.get_template('beam.pyx.tpl')
out = open("python/pydecode/potentials.pyx", "w")
print >>out, open("python/pydecode/templates/libhypergraph.pyx").read()
print >>out, template_beam.render(vars_beam)
print >>out, template.render(vars)


template = env.get_template('potentials.pxd.tpl')
template_beam = env.get_template('beam.pxd.tpl')
out = open("python/pydecode/potentials.pxd", "w")
print >>out, open("python/pydecode/templates/libhypergraph.pxd").read()
print >>out, template_beam.render(vars_beam)
print >>out, template.render(vars)
