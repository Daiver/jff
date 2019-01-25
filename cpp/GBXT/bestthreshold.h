#ifndef BESTTHRESHOLD
#define BESTTHRESHOLD

#include <algorithm>
#include <vector>
#include <iostream>

#include <Eigen/Core>

template<class Scalar, class GainScalar, typename GainFunc, typename Derived>
void findBestThreshold(
        const Eigen::MatrixBase<Derived> &values,
        const Eigen::Matrix<GainScalar, -1, 1> &gainValues,
        const GainFunc &gainFunc,
        double &bestGain,
        Scalar &bestThr)
{
    const int nSamples = values.rows();
    std::vector<int> indices(nSamples);
    for(int i = 0; i < indices.size(); ++i)
        indices[i] = i;
    std::sort(indices.begin(), indices.end(), 
        [&](const int a, const int b){return values[a] < values[b];});

    bestGain = -1e10;

    GainScalar Gl = 0;
    GainScalar Gr = 0;
    GainScalar Gall = 0;

    for(int sampleInd = 0; sampleInd < nSamples; ++sampleInd){
        Gall += gainValues[sampleInd];
    }
    Gr = Gall;

    for(int indInd = 0; indInd < nSamples - 1; ++indInd){
        const int sampleInd = indices[indInd];
        const GainScalar g = gainValues[sampleInd];
        Gr -= g;
        Gl += g;
        if(indInd < nSamples - 1){
            const int nextInd = indices[indInd + 1];
            const Scalar nextVal = values[nextInd];
            const Scalar curVal  = values[sampleInd];
            if(fabs(nextVal - curVal) < 0.0001){
                continue;
            }
        }
        Scalar curGain = gainFunc(Gall, Gl, Gr);
        if(curGain > bestGain){
            bestGain = curGain;
            auto thr = values[sampleInd];
            if(indInd < nSamples - 1){
                auto nextVal = values[indices[indInd + 1]];
                thr = (thr + nextVal)/2.0;
            }
            bestThr = thr;
        }
    }
}

#endif
