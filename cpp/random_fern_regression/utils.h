#ifndef UTILS_H
#define UTILS_H

#include <stdlib.h>

inline int randInt(const int start, const int finish)
{
    const int delta = finish - start;
    return (rand() % delta) + start;
}

inline float randomFloat(float a, float b) {
    float random = ((float) rand()) / (float) RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    return a + r;
}

#endif
