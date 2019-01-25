#include <iostream>
#include <vector>
#include <chrono>
#include <math.h>
#include "cpplinq.hpp"

std::chrono::high_resolution_clock::duration measure(std::function<void()> f, int n = 100)
{
    auto begin = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n; i++)
        f();
    auto end = std::chrono::high_resolution_clock::now();
    return (end - begin) / n;
}

const int nItems = 300000;

void simpleMap()
{
    std::vector<double> vec(nItems, 1);
    std::vector<double> res(nItems);
    for(int i = 0; i < nItems; ++i){
        //res.push_back(sin(vec[i])*2.0/5);
        res[i] = (sin(vec[i])*2.0/5);
    }
}

void linqMap()
{
    using namespace cpplinq;
    std::vector<double> vec(nItems, 1);
    std::vector<double> res = 
           from(vec) 
        >> select([](const double x)->double{return sin(x);})
        >> select([](const double x)->double{return (x)*2.0;})
        >> select([](const double x)->double{return (x)/5;})
        >> to_vector();
}

int main()
{
    auto simple_time = measure(simpleMap);
    std::cout << "simple " << simple_time.count()/10000000.0 << std::endl;
    auto linq_time = measure(linqMap);
    std::cout << "linq " << linq_time.count()/10000000.0 << std::endl;
    return 0;
}
