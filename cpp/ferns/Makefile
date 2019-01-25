CPP = clang++-3.5 -std=c++11 -O3
#CPP = g++ -std=c++11

bin/main: build/common.o build/fern.o
	$(CPP) -o bin/main main.cc build/*.o

build/common.o:
	$(CPP) -c -o build/common.o common.cpp

build/fern.o:
	$(CPP) -c -o build/fern.o fern.cpp
