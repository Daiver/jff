#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <vector>

namespace ml{

typedef std::vector<float> FeatureVec;
typedef std::vector<const FeatureVec *> DataSet;

DataSet datasetFrom2DArray(const std::vector<std::vector<float> > &arr);
int bestFreq(const std::vector<float> &freqs);

template<typename T>
void shuffle(std::vector<T> *vec, int iterations=-1);

template<typename A, typename B>
void shuffleBoth(std::vector<A> *vec1, std::vector<B> *vec2, int iterations=-1);

}


template<typename T>
void ml::shuffle(std::vector<T> *vec, int iterations)
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
void ml::shuffleBoth(std::vector<A> *vec1, std::vector<B> *vec2, int iterations)
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
#endif

