from build_config import *
from protoc import *
import os

build_mode = ""
if int(debug):
   env.Prepend(CCFLAGS =('-g',))
   build_mode = "debug/"
elif int(profile):
   env.Append(CCFLAGS = ('-O2', '-p', "-ggdb", "-fprofile-arcs", "-ftest-coverage", "-fno-strict-aliasing"),
              LINKFLAGS = ('-O2', '-p', "-ggdb" ,  "-fprofile-arcs", "-ftest-coverage", "-fno-strict-aliasing"))
   build_mode = "profile/"
else:
   env.Append(CCFLAGS = ('-O2', '-DNDEBUG', '-Werror', '-Wno-deprecated', "-fno-strict-aliasing"),
              LINKFLAGS = ('-O2', '-DNDEBUG', "-fno-strict-aliasing"))
   build_mode = "opt/"


for

env = Environment(CC = 'g++', ENV=os.environ, tools=['default', 'protoc', 'doxygen'], toolpath = ['.'])
env.Append(ROOT=build_config['root'])


variant = 'build/' + build_mode
env.VariantDir(variant, '.')

sub_dirs = ['#/' + variant + 'hypergraph',
            '#/' + variant + 'optimization']

libs = ('hypergraph', 'optimization', "protobuf", "pthread", "gflags")
lib_path = build_config['lib_extra']
include_path = build_config['include_extra']

env.Append(LIBPATH =('.',) + tuple(sub_dirs) + lib_path)

cpppath  = ('.', '#/third-party/svector/',
            '#/' + variant + 'interfaces/hypergraph/gen-cpp',
            '#/' + variant + 'interfaces/lattice/gen-cpp',
            '#/' + variant + 'interfaces/graph/gen-cpp') + \
            include_path + tuple(sub_dirs)

env.Append(CPPPATH=[cpppath])
env.Append(LIBS=libs)
env.Append(HYP_PROTO="#/" + variant + "interfaces/hypergraph/gen-cpp/")
env.Append(PROTOCPROTOPATH = [variant + "interfaces/graph/",
                              variant + "interfaces/hypergraph/"])

interfaces = SConscript(dirs=[variant + "interfaces"], exports=['env'])
local_libs = SConscript(dirs=sub_dirs, exports=['env', 'build_config'])

docs = env.Doxygen('Doxyfile')
env.Alias('document', docs)
