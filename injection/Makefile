all: inject_muons.cpp
	g++ inject_muons.cpp -o inject_muons -I/usr/include/hdf5/serial/ -I${PREFIX}/include/ -L${PREFIX}/lib/ -L${PREFIX}/lib64/ -L${PREFIX}/lib/pkgconfig/ -lLeptonInjector -g
