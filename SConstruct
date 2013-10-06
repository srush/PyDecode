import os


for build_mode in ['debug', 'profile', 'opt']:
    env = Environment(CC = 'g++', ENV=os.environ)

    # conf = Configure(env)
    # if not conf.CheckLib('boost_system'):
    #     print 'Did not find libboost, exiting!'
    #     Exit(1)
    # env = conf.Finish()


    if build_mode == "debug":
        env.Prepend(CCFLAGS =('-g', '-fPIC', '-Wall'))
    elif build_mode == "profile":
        env.Append(CCFLAGS = ('-O2', '-p', "-ggdb",
                              "-fprofile-arcs", "-ftest-coverage",
                              "-fno-strict-aliasing"),
                   LINKFLAGS = ('-O2', '-p', "-ggdb" ,
                                "-fprofile-arcs",
                                "-ftest-coverage",
                                "-fno-strict-aliasing"))
    elif build_mode == "opt":
        env.Append(CCFLAGS = ('-O2', '-DNDEBUG',  '-fPIC',
                              '-Werror', '-Wno-deprecated',
                              "-fno-strict-aliasing"),
                   LINKFLAGS = ('-O2', '-DNDEBUG', '-fPIC',
                                "-fno-strict-aliasing"))

    variant = 'build/' + build_mode + "/"
    env.VariantDir(variant, '.')
    sub_dirs = ['#/' + variant + 'src']
    libs = ('decoding')
    env.Append(LIBPATH =('.',) + tuple(sub_dirs))

    cpppath = ('.', '#/' + variant + 'interfaces/hypergraph/gen-cpp',
               tuple(sub_dirs))

    env.Append(CPPPATH=[cpppath])
    env.Append(LIBS=libs)

    #interfaces = env.SConscript(dirs=[variant + "interfaces"], exports=['env'])
    local_libs = env.SConscript(dirs=sub_dirs, exports=['env'])
    #env.Program(variant + 'run', variant + "src/run.cpp", LIBS = libs)
