#ifndef BESTSPLIT_H
#define BESTSPLIT_H

#include <vector>

#include "bestthreshold.h"

template<
    class Scalar, 
    class GainScalar,
    class GainFunc>
void findBestSplit(
        const Eigen::Matrix<Scalar, -1, -1> &data,
        const Eigen::Matrix<GainScalar, -1, 1> &gainValues,
        const std::vector<int> &dataIndices,
        const std::vector<int> &featIndices,
        const GainFunc &gainFunc,
        double &bestGain,
        int    &bestFeatInd,
        Scalar &bestThr)
{
    Eigen::Matrix<Scalar, -1, 1>     dataLocal(dataIndices.size());
    Eigen::Matrix<GainScalar, -1, 1> gvaluesLocal(dataIndices.size());

    for(int i = 0; i < gvaluesLocal.rows(); ++i)
        gvaluesLocal[i] = gainValues[dataIndices[i]];

    bestGain = -1e10;
    for(int featIndInd = 0; featIndInd < featIndices.size(); ++featIndInd){
        const int featInd = featIndices[featIndInd];

        for(int i = 0; i < gvaluesLocal.rows(); ++i)
            dataLocal[i] = data(dataIndices[i], featInd);
       
        double curGain;
        Scalar curThr;
        findBestThreshold<Scalar, GainScalar>(
                dataLocal,
                gvaluesLocal,
                gainFunc,
                curGain, curThr);

        if(curGain > bestGain){
            bestGain = curGain;
            bestThr  = curThr;
            bestFeatInd = featInd;
        }
    }
}

#endif
