import os

for build_mode in ['debug', 'profile', 'opt']:
    env = Environment(CC = 'g++', ENV=os.environ)

    if build_mode == "debug":
        env.Prepend(CCFLAGS =('-g', '-fPIC', '-Wall', '-std=c++11'))
    elif build_mode == "profile":
        env.Append(CCFLAGS = ('-O2', '-p', "-ggdb",
                              "-fprofile-arcs", "-ftest-coverage",
                              "-fno-strict-aliasing", '-std=c++11'),
                   LINKFLAGS = ('-O2', '-p', "-ggdb" ,
                                "-fprofile-arcs",
                                "-ftest-coverage",
                                "-fno-strict-aliasing"))
    elif build_mode == "opt":
        env.Append(CCFLAGS = ('-O2', '-fPIC',
                              '-Werror', '-Wno-deprecated',
                              "-fno-strict-aliasing", '-std=c++11'),
                   LINKFLAGS = ('-O2', '-fPIC',
                                "-fno-strict-aliasing"))

    variant = 'build/' + build_mode + "/"
    env.VariantDir(variant, '.')
    sub_dirs = ['#/' + variant + 'src']
    libs = ('decoding')
    env.Append(LIBPATH =('.',) + tuple(sub_dirs))
    cpppath = ('.', tuple(sub_dirs))
    env.Append(CPPPATH=[cpppath])
    env.Append(LIBS=libs)
    local_libs = env.SConscript(dirs=sub_dirs, exports=['env'])
