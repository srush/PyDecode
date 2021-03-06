import os

local_libs = {}
for build_mode in ['debug', 'profile', 'opt']:
    env = Environment(CXX="g++", ENV=os.environ)

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
        env.Append(CCFLAGS = ('-O2', '-fPIC',
                              '-Werror', '-Wno-deprecated',
                              "-fno-strict-aliasing"),
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
    local_libs[build_mode] = env.SConscript(dirs=sub_dirs, exports=['env'])

# Run the C++ tests.
env = Environment(ENV=os.environ, CXX="g++")
env.Append(CPPPATH = ["src/"], LIBPATH = ['/usr/lib/', '/usr/local/lib'])

b = env.Program("build/test", 'src/Tests.cpp',
                LIBS = ["pthread", "gtest"] + local_libs["debug"])

b2 = env.Command("build/test.out", b, "build/test")
env.Alias("test", b2)


# Build the docs.
# notebooks = env.Command("ignore_note", [], "cd notebooks;make all")
# env.AlwaysBuild(notebooks)

# doxygen = env.Command("ignore_dox", [], "doxygen Doxyfile")
# env.AlwaysBuild(doxygen)

docs = env.Command("ignore_docs", [], "cd docs; make html")
env.AlwaysBuild(docs)

env.Alias("docs", [docs])

# Run the python tests.
pytests = env.Command("ignore_test", [], "nosetests python/pydecode/test/")
env.AlwaysBuild(pytests)

pytests2 = env.Command("ignore_test2", [], "py.test notebooks")
env.AlwaysBuild(pytests)

env.Alias("pytest", [pytests, pytests2])



# Building the python library.

env.Command(["python/pydecode/_pydecode.pyx",],
            ["python/pydecode/templates/algorithms.pyx.tpl",
             "python/pydecode/templates/algorithms.pxd.tpl",
             "python/pydecode/templates/chart.pyx",
             "python/pydecode/templates/beam.pyx.tpl",
             "python/pydecode/templates/cython_jinja.py"],
            "python python/pydecode/templates/cython_jinja.py")

py_lib = env.Command(["python/pydecode/_pydecode.so"],
            ["build/debug/src/libdecoding.a",
             "python/pydecode/_pydecode.pyx",
             "python/pydecode/templates/libhypergraph.pyx"],
            "CC='ccache g++' python setup.py build_ext --inplace --verbose --cython --debug")
env.Alias("pylib", [py_lib])
