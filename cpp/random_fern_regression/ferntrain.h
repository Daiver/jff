#ifndef RANDOMFERNTRAIN_H
#define RANDOMFERNTRAIN_H

//#include <climits>
#include <iostream>
#include <stdlib.h>
#include "utils.h"
#include "fern.h"

namespace RandomFern {

template<typename Derived>
inline void sampleFeatInds(const int nFeats, Eigen::MatrixBase<Derived> &vec)
{
    assert(vec.cols() == 1);
    for(int i = 0; i < vec.rows(); ++i){
        const int index = randInt(0, nFeats);
        vec(i, 0) = index;
    }
}

template<typename Derived1, typename Derived2>
inline void sampleThresholds(
        const Eigen::Matrix<std::pair<float, float>, -1, 1> &borders, 
        const Eigen::MatrixBase<Derived1> &featInds,
        Eigen::MatrixBase<Derived2> &vec)
{
    assert(vec.cols() == 1);
	for(int i = 0; i < vec.rows(); ++i){
        const int featInd = featInds[i];
		vec(i, 0) = randomFloat(borders(featInd, 0).first, borders(featInd, 0).second);
    }
}

template<int Depth, typename Output>
inline void fillBins(
        const Eigen::MatrixXf &data,
        const Eigen::Matrix<Output, -1, 1> &targets,
        Fern<Depth, Output> &fern)
{
    const int nSamples = data.rows();
    for(int i = 0; i < fern.nBins; ++i)
        fern.bins[i] = Output::Zero();
    Eigen::Matrix<int, fern.nBins, 1> counts = Eigen::Matrix<int, fern.nBins, 1>::Zero();
    for(int sampleInd = 0; sampleInd < nSamples; ++sampleInd){
        const int binInd = fern.getIndex(data.row(sampleInd));
        counts[binInd] += 1;
        fern.bins[binInd] += targets[sampleInd];
    }
    for(int i = 0; i < fern.nBins; ++i)
        if(counts[i] > 0)
            fern.bins[i] /= counts[i];
}

template<int Depth, typename Output>
inline void trainOneFern(
        const Eigen::Matrix<std::pair<float, float>, -1, 1> &borders, 
        const Eigen::MatrixXf &data,
        const Eigen::Matrix<Output, -1, 1> &targets,
        Fern<Depth, Output> &fern)
{
    sampleFeatInds(data.cols(), fern.featInds);
    sampleThresholds(borders, fern.featInds, fern.thresholds);
    fillBins(data, targets, fern);
}

template<int Depth, typename Output>
double scoreFern(
        const Eigen::MatrixXf &data,
        const Eigen::Matrix<Output, -1, 1> &targets,
        const Fern<Depth, Output> &fern)
{
    double res = 0;
    const int nSamples = data.rows();
    for(int i = 0; i < nSamples; ++i){
        //const Eigen::VectorXf sample = data.row(i);
        const Output diff = fern.predict(data.row(i)) - targets(i, 0);
        res += diff.norm();
    }
    return res;
}

template<int Depth, typename Output>
Fern<Depth, Output> trainFern(
        const Eigen::Matrix<std::pair<float, float>, -1, 1> &borders, 
        const Eigen::MatrixXf &data,
        const Eigen::Matrix<Output, -1, 1> &targets,
        const int nRepeats)
{
    typedef Fern<Depth, Output> FernRes;
    FernRes bestFern;
    FernRes curFern;
    double bestErr = 1e12;
    for(int iter = 0; iter < nRepeats; ++iter){
        trainOneFern(borders, data, targets, curFern);
        //bestFern = curFern;
        const double err = scoreFern(data, targets, curFern);
        if(err < bestErr){
            bestErr = err;
            bestFern = curFern;
        }
    }
    //std::cout << "bestErr " << bestErr << std::endl;
    return bestFern;
}

}

#endif
