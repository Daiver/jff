#include "fern.h"
#include <stdio.h>

float getPixel(const std::vector<float> &features, int width, const Point &p)
{
    return features[p.i*width + p.j];
}

int getIndex(
       const std::vector<float> &features, int width, const Fern &fern)
{
    int index = 0;
    for(int j = 0; j < fern.points.size(); ++j){
        index *= 2;
        if(   getPixel(features, width, fern.points[j].first) 
            < getPixel(features, width, fern.points[j].second))
            index += 1;
    }
    return index;
}

std::vector<float> getProbs(
    const std::vector<float> &features,
    int width,
    std::vector<Fern> &ferns, int countOfClasses)
{
    std::vector<float> res(countOfClasses, 0);

    for(int k = 0; k < ferns.size(); ++k){
        int index = getIndex(features, width, ferns[k]);
        for(int i = 0; i < countOfClasses; ++i)
            res[i] += ferns[k].probs[index][i];
    }

    return res;
}

void trainFern(
        const std::vector<const std::vector<float> *> &dataSet,
        int width, int height,
        const std::vector<int> &labels,
        int countOfClasses,
        int countOfFeaturesPerFern,
        Fern &fern)
{
    fern.points.resize(countOfFeaturesPerFern);
    for(int i = 0; i < pow(2, countOfFeaturesPerFern); ++i)
        fern.probs.push_back(std::vector<float>(countOfClasses, 0.0));

    float inverseProb = 1.0/countOfClasses;

    for(int i = 0; i < countOfFeaturesPerFern; ++i){
        Point p1 (random() % width, random() % height);
        Point p2 (random() % width, random() % height);
        fern.points[i].first  = p1;
        fern.points[i].second = p2;
    }

    const float u = 1.0;
    std::vector<float> theta(pow(2, countOfFeaturesPerFern), 0);
    //std::vector<int>   counts(pow(2, countOfFeaturesPerFern), 0);
    std::vector<int>   counts(countOfClasses, 0);
    //float pEvent = dataSet.size()/(dataSet.size() + u*countOfClasses);

    for(int i = 0; i < 300000; ++i){
    //for(int i = 0; i < dataSet.size(); ++i){
        int sampleIndex = random() % dataSet.size();
        int index = getIndex(*dataSet[sampleIndex], width, fern);
        theta[index] += 1;
        counts[labels[sampleIndex]] += 1;
        fern.probs[index][labels[sampleIndex]] += 1;
    }
    
    for(int i = 0; i < theta.size(); ++i)
        theta[i] = ((float)theta[i]/(theta[i] + u*countOfClasses));

    for(int i = 0; i < fern.probs.size(); ++i){
        for(int j = 0; j < fern.probs[i].size(); ++j){
            //if(counts[i] == 0)
            //    continue;
            fern.probs[i][j] = log((fern.probs[i][j] + u)/(counts[j] + u * pow(2, countOfFeaturesPerFern)));
/*            fern.probs[i][j] = (*/
                    //((u + fern.probs[i][j]) / 
                        //(counts[j] + u * pow(2, countOfFeaturesPerFern)))*theta[i] 
                    /*+ inverseProb * (1.0 - theta[i]));*/
            //printf("%d %d %f %f\n", i, j, fern.probs[i][j], theta[i]);
        }
    }
}
