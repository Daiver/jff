#ifndef FERN_H
#define FERN_H

#include <math.h>
#include <stdlib.h>

#include "common.h"

struct Point
{
    Point(){}

    Point(int i, int j): i(i), j(j) {}

    int i;
    int j;
};

class Fern
{
public:
    std::vector<std::vector<float>> probs;
    std::vector<std::pair<Point, Point>> points;

};

void trainFern(
        const std::vector<const std::vector<float> *> &dataSet,
        int width, int height,
        const std::vector<int> &labels,
        int countOfClasses,
        int countOfFeaturesPerFern,
        Fern &fern);


std::vector<float> getProbs(
        const std::vector<float> &features, int width,
        std::vector<Fern> &ferns, int countOfClasses);

#endif
