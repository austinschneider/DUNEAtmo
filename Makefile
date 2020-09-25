all: inject_muons.cpp
	g++ inject_muons.cpp -o inject_muons -I/usr/include/hdf5/serial/ -I${GOLEMBUILDPATH}/include/ -L${GOLEMBUILDPATH}/lib/ -L${GOLEMBUILDPATH}/lib/pkgconfig/ -lLeptonInjector -g
