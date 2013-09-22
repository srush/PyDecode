from build_config import *
from protoc import *
import os



for build_mode in ['debug', 'profile', 'opt']:
    env = Environment(CC = 'g++', ENV=os.environ, tools=['default', 'protoc', 'doxygen'], toolpath = ['.'])
    env.Append(ROOT=build_config['root'])

    if build_mode == "debug":
        env.Prepend(CCFLAGS =('-g', '-fPIC'))
    elif build_mode == "profile":
        env.Append(CCFLAGS = ('-O2', '-p', "-ggdb", "-fprofile-arcs", "-ftest-coverage", "-fno-strict-aliasing"),
                   LINKFLAGS = ('-O2', '-p', "-ggdb" ,  "-fprofile-arcs", "-ftest-coverage", "-fno-strict-aliasing"))
    elif build_mode == "opt":
        env.Append(CCFLAGS = ('-O2', '-DNDEBUG', '-Werror', '-Wno-deprecated', "-fno-strict-aliasing"),
                   LINKFLAGS = ('-O2', '-DNDEBUG', "-fno-strict-aliasing"))

    variant = 'build/' + build_mode + "/"
    env.VariantDir(variant, '.')
    sub_dirs = ['#/' + variant + 'src']

    #libs = ('decoding', "protobuf", "pthread", "gflags")
    libs = ('decoding')
    include_path = build_config['include_extra']

    env.Append(LIBPATH =('.',) + tuple(sub_dirs))

    cpppath = ('.', '#/' + variant + 'interfaces/hypergraph/gen-cpp',
               include_path + tuple(sub_dirs))

    env.Append(CPPPATH=[cpppath])
    env.Append(LIBS=libs)
    env.Append(HYP_PROTO="#/" + variant + "interfaces/hypergraph/gen-cpp/")
    env.Append(HYP_PROTO_PY="#/python/decoding/interfaces/")
    env.Append(PROTOCPROTOPATH = [variant + "interfaces"])
    #interfaces = env.SConscript(dirs=[variant + "interfaces"], exports=['env'])
    local_libs = env.SConscript(dirs=sub_dirs, exports=['env', 'build_config'])
    env.Program(variant + 'run', variant + "src/run.cpp", LIBS = libs)
    #env.Alias('proto', interfaces)
# docs = env.Doxygen('Doxyfile')
# env.Alias('document', docs)
