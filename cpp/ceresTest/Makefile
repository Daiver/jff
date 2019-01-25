LINK = /usr/bin/c++      regression1.cpp.o  -o regression1 -rdynamic /home/daiver/coding/ceres-bin/install/lib/libceres.a -lglog /usr/local/lib/libspqr.a -ltbb -ltbbmalloc /usr/local/lib/libcholmod.a /usr/local/lib/libccolamd.a /usr/local/lib/libcamd.a /usr/local/lib/libcolamd.a /usr/local/lib/libamd.a -llapack -lf77blas -latlas /usr/local/lib/libsuitesparseconfig.a -lrt /usr/local/lib/libcxsparse.a -llapack -lf77blas -latlas /usr/local/lib/libsuitesparseconfig.a -lrt /usr/local/lib/libcxsparse.a -lgomp -lpthread 

regression1:
	g++ -c regression1.cpp -o regression1.cpp.o -I/home/daiver/coding/ceres-bin/install/include -I/usr/local/include/eigen3
	$(LINK)
