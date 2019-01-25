#include "ferns.h"

void trainCascade(
        const DataSet &dataSet, 
        const std::vector<float> &initialValues, 
        int depthOfFern, 
        int fernsPoolSize,
        int cascadeSize,
        std::vector<Fern> &cascade)
{
    auto ranges = computeRangesOfFeatures(dataSet);

/*    for(int i = 0; i < ranges.size(); ++i)*/
        /*printf("R %f %f\n", ranges[i].first, ranges[i].second);*/

    cascade.reserve(cascadeSize);
    //DataSet dataSet = initialDataSet;
    std::vector<float> values = initialValues;
    
    Fern *curFern = new Fern(depthOfFern);
    Fern *bestFern = new Fern(depthOfFern);
    for(int outterIter = 0; outterIter < cascadeSize; ++outterIter){
        float bestErr = FLT_MAX;
        for(int innerIter = 0; innerIter < fernsPoolSize; ++innerIter){
            curFern->fit(dataSet, values, ranges);
            float err = curFern->evalError(dataSet, values);
            if(err < bestErr){
                bestErr = err;
                //delete bestFern;
                Fern *tmp = curFern;
                curFern = bestFern;
                bestFern = tmp;
            }
        }

        for(int i = 0; i < values.size(); ++i)
            values[i] -= bestFern->activate(dataSet[i]);
        cascade.push_back(*bestFern);
        printf("%d stage finished bestErr: %f\n", outterIter, bestErr);
    }
    delete bestFern;
    delete curFern;
}

float activateCascade(const std::vector<Fern> &cascade, const FeatureVector &vec)
{
    float res = 0;
    for(auto &&fern : cascade)
        res += fern.activate(vec);
    return res;
}
