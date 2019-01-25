CPP = clang++-3.5
#CPP = g++
FLAGS = -O3 -std=c++11

bin/main: bin/tests
	$(CPP) $(FLAGS) -o bin/main main.cpp build/*.o

bin/tests: build/ferns.o build/common.o
	$(CPP) $(FLAGS) -o bin/tests tests.cpp build/*.o

build/ferns.o:
	$(CPP) $(FLAGS) -c -o build/ferns.o ferns.cpp

build/common.o:
	$(CPP) $(FLAGS) -c -o build/common.o common.cpp
