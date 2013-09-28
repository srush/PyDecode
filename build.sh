cython --cplus python/decoding/binding.pyx
scons build/debug/src/libdecoding.a
rm -fr build/temp.linux-x86_64-2.7/
rm -fr build/lib.linux-x86_64-2.7/
python setup.py clean
python setup.py config
python setup.py build
sudo python setup.py install
