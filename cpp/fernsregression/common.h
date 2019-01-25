#ifndef COMMON_H
#define COMMON_H

#include <vector>
#include <string>
#include <sstream>
#include <stdlib.h>
#include <cfloat>
#include <iostream>

typedef std::vector<float> FeatureVector;
typedef std::vector<FeatureVector> DataSet;

typedef std::pair<float, float> Range;

std::vector<Range> computeRangesOfFeatures(const DataSet &dataSet);

template<typename T>
void shuffle(std::vector<T> *vec, int iterations)
{
    if(iterations == -1)
        iterations = vec->size();
    for(int i = 0; i < iterations; ++i){
        int ind1 = rand() % vec->size();
        int ind2 = rand() % vec->size();
        T c = (*vec)[ind1];
        (*vec)[ind1] = (*vec)[ind2];
        (*vec)[ind2] = c;
    }
}

template<typename A, typename B>
void shuffleBoth(std::vector<A> *vec1, std::vector<B> *vec2, int iterations)
{
    if(iterations == -1)
        iterations = vec1->size();
    for(int i = 0; i < iterations; ++i){
        int ind1 = rand() % vec1->size();
        int ind2 = rand() % vec1->size();
        A c1 = (*vec1)[ind1];
        (*vec1)[ind1] = (*vec1)[ind2];
        (*vec1)[ind2] = c1;

        B c2 = (*vec2)[ind1];
        (*vec2)[ind1] = (*vec2)[ind2];
        (*vec2)[ind2] = c2;
    }
}

std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems);

std::vector<std::string> split(const std::string &s, char delim);

//void splitDataSetByColumn(const DataSet &ds, int columnIndex

template<typename T>
void printVecHor(const std::vector<T> &vec)
{
    for(auto &&x : vec)
        std::cout << x << " ";
    std::cout << std::endl;
}

template<typename T>
void printVec2D(const std::vector<std::vector<T>> &vec)
{
    for(auto &&x : vec)
        printVecHor(x);
}


#endif
