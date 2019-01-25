#include "stumptests.h"

#include <range/v3/all.hpp>

#include "stump.h"
#include "targetlossinfomse.h"

StumpTests::StumpTests()
{

}

void StumpTests::testSortIndicesByFeature01()
{
    Eigen::MatrixXf data(3, 3);
    data << 1, 2, 3,
            4, 5, 6,
            7, 8, 9;
    const int featInd = 1;
    std::vector<long> indices = {0, 1, 2};
    Stump::sortIndicesByFeature(data, featInd, indices);
    std::vector<long> ans = {0, 1, 2};
    QCOMPARE(indices, ans);
}

void StumpTests::testSortIndicesByFeature02()
{
    Eigen::MatrixXf data(3, 3);
    data << 1, 2, 3,
            4, 5, 6,
            7, 8, 9;
    const int featInd = 1;
    std::vector<long> indices = {2, 0};
    Stump::sortIndicesByFeature(data, featInd, indices);
    std::vector<long> ans = {0, 2};
    QCOMPARE(indices, ans);
}

void StumpTests::testSortIndicesByFeature03()
{
    Eigen::Matrix3f data;
    data << 1, 2, 6,
            4, 5, 2,
            7, 8, 3;
    const int featInd = 2;
    std::vector<long> indices = {0, 1, 2};
    Stump::sortIndicesByFeature(data, featInd, indices);
    std::vector<long> ans = {1, 2, 0};
    QCOMPARE(indices, ans);
}

void StumpTests::testTargetLossInfoMSE01()
{
    Eigen::VectorXf targets(5);
    targets << 1, 2, 3, 4, 5;

    Stump::TargetsLossInfoMSE lossInfo = Stump::TargetsLossInfoMSE::fromTargets(targets, ranges::view::ints(0) | ranges::view::take(5));
    QVERIFY(fabs(3.0 - lossInfo.mean()) < 0.00001);
    auto loss = lossInfo.loss();
    QVERIFY(fabs(2.0 - loss) < 0.00001);
}

void StumpTests::testTargetLossInfoMSE02()
{
    Eigen::VectorXf targets(7);
    targets << 30, 10000, 20, 40, -5, 6, 12;

    Stump::TargetsLossInfoMSE lossInfo = Stump::TargetsLossInfoMSE::fromTargets(targets, {0, 2, 3, 4, 5, 6});
    auto mean = lossInfo.mean();
    QVERIFY(fabs(17.166666666666668 - mean) < 0.001);
    auto loss = lossInfo.loss();
    QVERIFY(fabs(222.80555550257364 - loss) < 0.001);
}

void StumpTests::testTargetLossInfoMSE03()
{
    Eigen::VectorXf targets(5);
    targets << 1, 2, 3, 4, 5;

    Stump::TargetsLossInfoMSE lossInfo = Stump::TargetsLossInfoMSE::fromTargets(targets, ranges::view::ints(0) | ranges::view::take(5));
    lossInfo.removeItem(5);

    auto mean = lossInfo.mean();
    QVERIFY(fabs(2.5 - mean) < 0.001);
    auto loss = lossInfo.loss();
    QVERIFY(fabs(1.25 - loss) < 0.001);

    lossInfo.addItem(5);
    mean = lossInfo.mean();
    QVERIFY(fabs(3 - mean) < 0.001);
    loss = lossInfo.loss();
    QVERIFY(fabs(2 - loss) < 0.001);
}

void StumpTests::testFindBestSplitAcrossFeature01()
{
    Eigen::MatrixXf data(5, 1);
    data << 1, 2, 3, 4, 5;
    Eigen::VectorXf targets(5);
    targets << 1, 2, 3, 4, 5;

    const std::vector<long> samplesIndices = ranges::view::ints(0) | ranges::view::take(data.rows());
    const int featInd = 0;

    auto res = Stump::findBestSplitAcrossFeature(data, targets, samplesIndices, featInd);
    QCOMPARE(res.splitInfo.featInd, 0);
    QVERIFY(fabs(res.splitInfo.threshold - 2.5) < 0.00001);
    QVERIFY(fabs(res.score - 0.453) < 0.01);
}
