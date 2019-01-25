#include "stump.h"
#include <algorithm>

#include "targetlossinfomse.h"

Stump::SplittingResult Stump::findBestSplitAcrossFeature(
        const Eigen::MatrixXf &data,
        const Eigen::VectorXf &targets,
        const std::vector<long> &samplesIndicesInit,
        const int featInd)
{
    auto samplesIndicesSorted = sortIndicesByFeature(data, samplesIndicesInit, featInd);

    TargetsLossInfoMSE lossInfoLeft;
    TargetsLossInfoMSE lossInfoRight = TargetsLossInfoMSE::fromTargets(targets, samplesIndicesSorted);

    float bestLoss = lossInfoRight.loss();
    float bestThreshold = targets[0];
    const int nIndices = samplesIndicesSorted.size();
    for(int indOfInd = 0; indOfInd < nIndices - 1; ++indOfInd){
        const int index = samplesIndicesSorted[indOfInd];
        const auto target = targets[index];
        lossInfoRight.removeItem(target);
        lossInfoLeft.addItem(target);

        const float curLoss = 0.5 * (lossInfoRight.loss() + lossInfoLeft.loss());
        if(curLoss < bestLoss){
            bestLoss = curLoss;
            const float threshold = 0.5 * (data(samplesIndicesSorted[indOfInd + 1], featInd) + data(index, featInd));
            bestThreshold = threshold;
        }
    }

    return SplittingResult{bestLoss, SplittingInfo{featInd, bestThreshold}};
}


