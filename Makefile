all: inject_muons.cpp inject_muons_sk.cpp
	g++ inject_muons.cpp -o inject_muons -I/usr/include/hdf5/serial/ -I${GOLEMBUILDPATH}/include/ -L${GOLEMBUILDPATH}/lib/ -L${GOLEMBUILDPATH}/lib/pkgconfig/ -lLeptonInjector -g
	g++ inject_muons_sk.cpp -o inject_muons_sk -I/usr/include/hdf5/serial/ -I${GOLEMBUILDPATH}/include/ -L${GOLEMBUILDPATH}/lib/ -L${GOLEMBUILDPATH}/lib/pkgconfig/ -lLeptonInjector -g
	g++ inject_muons_volume_dune.cpp -o inject_muons_volume_dune -I/usr/include/hdf5/serial/ -I${GOLEMBUILDPATH}/include/ -L${GOLEMBUILDPATH}/lib/ -L${GOLEMBUILDPATH}/lib/pkgconfig/ -lLeptonInjector -g
	g++ inject_muons_small.cpp -o inject_muons_small -I/usr/include/hdf5/serial/ -I${GOLEMBUILDPATH}/include/ -L${GOLEMBUILDPATH}/lib/ -L${GOLEMBUILDPATH}/lib/pkgconfig/ -lLeptonInjector -g
