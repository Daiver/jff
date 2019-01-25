#ifndef FERNS_H
#define FERNS_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <string>

#include "common.h"

class Fern
{
public:
    Fern(int depth)
    {
        featsIndices.resize(depth);
        featsBarriers.resize(depth);
        regressionValues.resize(pow(2, depth));
    }

    std::vector<int>   featsIndices;
    std::vector<float> featsBarriers;
    std::vector<float> regressionValues;

    int depth() const 
    {
        return featsIndices.size();
    }

    int getIndex(const FeatureVector &vec) const
    {
        int index = 0;
        for(int i = 0; i < featsIndices.size(); ++i){
            index *= 2;
            if(vec[featsIndices[i]] >= featsBarriers[i])
                index += 1;
        }
        return index;
    }

    float activate(const FeatureVector &vec) const
    {
        return regressionValues[getIndex(vec)];
    }

    float evalError(const DataSet &ds, const std::vector<float> &values) const
    {
        float err = 0;
        for(int i = 0; i < values.size(); ++i)
            err += fabs(values[i] - activate(ds[i]));
        return err/ds.size();
    }

    void fit(const DataSet &ds, const std::vector<float> &values, const std::vector<Range> &ranges)
    {
/*        featsIndices.fill(featsIndices.begin(), featsIndices.end(), -1);*/
        //featsBarriers.fill(0);
        /*regressionValues.fill(0);*/
        for(int i = 0; i < featsIndices.size(); ++i){
            int index = random() % ds[0].size();
            featsIndices[i] = index;
            featsBarriers[i] = ranges[index].first + random() / (float)RAND_MAX * 
                                (ranges[index].second - ranges[index].first);
        }
        
        std::vector<int> counts(pow(2, depth()), 0);

        for(int i = 0; i < regressionValues.size(); ++i)
            regressionValues[i] = 0;

        for(int i = 0; i < ds.size(); ++i){
            auto &&sample = ds[i];
            int index = getIndex(sample);
            //printf("Index %d\n", index);
            regressionValues[index] += values[i];
            counts[index]++;
        }

        for(int i = 0; i < regressionValues.size(); ++i){
            if(counts[i] == 0)
                continue;
            const float beta = 1.0;
            regressionValues[i] /= (1 + beta/counts[i]) * counts[i];
            //regressionValues[i] /= counts[i];
        }
    }

    void print()
    {
        printf("Fern(bars:");
        for(int i = 0; i < featsIndices.size(); ++i)
            printf("%d:%f ", featsIndices[i], featsBarriers[i]);
        printf(" vals:");
        for(int i = 0; i < regressionValues.size(); ++i)
            printf("%f ", regressionValues[i]);
        printf(")\n");
    }
};


void trainCascade(
        const DataSet &dataSet, 
        const std::vector<float> &values, 
        int depthOfFern, 
        int fernsPoolSize,
        int cascadeSize,
        std::vector<Fern> &cascade);

float activateCascade(const std::vector<Fern> &cascade, const FeatureVector &vec);

#endif
