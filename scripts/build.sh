cython --cplus python/decoding/binding.pyx
scons build/opt/src/libdecoding.a
rm -fr build/temp.linux-x86_64-2.7/
rm -fr build/lib.linux-x86_64-2.7/
rm -fr python/pydecode/hyper.so
python setup.py clean
python setup.py config
python setup.py build
python setup.py build_ext --inplace
sudo python setup.py install
