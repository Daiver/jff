
#CPP = g++
CPP = clang++-3.5

CPP_FLAGS = -std=c++14 -Wno-deprecated-register -Wall -Wpedantic

main:
	$(CPP) $(CPP_FLAGS) -o main main.cpp -I/usr/include/eigen3/

tests:
	$(CPP) $(CPP_FLAGS) -o tests tests.cpp -I/usr/include/eigen3/ && ./tests

