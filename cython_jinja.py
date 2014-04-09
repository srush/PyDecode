from jinja2 import Environment, PackageLoader
import yaml


env = Environment(loader=PackageLoader('pydecode', 'templates'))

vars = yaml.load(open("python/pydecode/templates/potentials.yaml"))

for var in vars["semirings"]:
    var["ctype"] = var["type"] + "Potential"

template = env.get_template('potentials.pyx.tpl')
out = open("python/pydecode/potentials.pyx", "w")
print >>out, template.render(vars)
print >>out, open("python/pydecode/libhypergraph.pyx").read()

template = env.get_template('potentials.pxd.tpl')
out = open("python/pydecode/potentials.pxd", "w")
print >>out, open("python/pydecode/libhypergraph.pxd").read()

print >>out, template.render(vars)
