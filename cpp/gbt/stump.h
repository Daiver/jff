#ifndef STUMP_H
#define STUMP_H

#include <vector>
#include "Eigen/Core"
#include "splittinginfo.h"

namespace Stump {

class SplittingResult
{
public:
    float score;
    SplittingInfo splitInfo;
};

template<typename Derived>
void sortIndicesByFeature(
        const Eigen::MatrixBase<Derived> &data,
        const int featInd,
        std::vector<long> &samplesIndices);

template<typename Derived>
std::vector<long> sortIndicesByFeature(
        const Eigen::MatrixBase<Derived> &data,
        const std::vector<long> &samplesIndices,
        const int featInd);

SplittingResult findBestSplitAcrossFeature(
        const Eigen::MatrixXf &data,
        const Eigen::VectorXf &targets,
        const std::vector<long> &samplesIndices,
        const int featInd);

}















//IMPLEMENTATIONS

template<typename Derived>
inline void Stump::sortIndicesByFeature(
        const Eigen::MatrixBase<Derived> &data,
        const int featInd,
        std::vector<long> &samplesIndices)
{
    assert(samplesIndices.size() > 0 && samplesIndices.size() <= (size_t)data.rows());
    std::sort(samplesIndices.begin(), samplesIndices.end(), [&](const long &ind1, const long &ind2) -> bool {
        return data(ind1, featInd) < data(ind2, featInd);
    });
}

template<typename Derived>
inline std::vector<long> Stump::sortIndicesByFeature(
        const Eigen::MatrixBase<Derived> &data,
        const std::vector<long> &samplesIndices,
        const int featInd)
{
    std::vector<long> res = samplesIndices;
    sortIndicesByFeature(data, featInd, res);
    return res;
}


#endif // STUMP_H
