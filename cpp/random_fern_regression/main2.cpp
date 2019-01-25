#include <iostream>

#include "Eigen/Core"
#include "fern.h"
#include "ferntrain.h"
#include "boosting.h"
#include "eigexport.h"

using namespace RandomFern;

Eigen::Matrix<float, 1, 1> toEig(const float f)
{
    Eigen::Matrix<float, 1, 1> res;
    res[0] = f;
    return res;
}

typedef Eigen::Matrix<float, 1, 1> Output;

void genCircleData(const int nPoints, Eigen::MatrixXf &data, Eigen::Matrix<Output, -1, 1> &targets)
{
    const float low  = -1.2;
    const float high =  1.2;
    const float radius = 1.0;

    data.resize(nPoints, 2);
    targets.resize(nPoints, 1);

    for(int i = 0; i < nPoints; ++i){
        const float x = randomFloat(low, high);
        const float y = randomFloat(low, high);
        data(i, 0) = x;
        data(i, 1) = y;
        if(x*x + y*y >= radius*radius)
            targets[i] = toEig(0.0);
        else
            targets[i] = toEig(1.0);
    }
}

Eigen::Matrix<std::pair<float, float>, -1, 1> computeBorders(const Eigen::MatrixXf &data)
{
    Eigen::Matrix<std::pair<float, float>, -1, 1> res(data.cols());
    const Eigen::VectorXf maxs = data.array().colwise().maxCoeff();
    const Eigen::VectorXf mins = data.array().colwise().minCoeff();
    for(int i = 0; i < data.cols(); ++i){
        res(i, 0) = std::make_pair(mins[i], maxs[i]);
    }
    return res;
}

int main()
{
    srand(42);
    const int nPoints = 100000;
    Eigen::MatrixXf data;
    Eigen::Matrix<Output, -1, 1> targets;
    genCircleData(nPoints, data, targets);
    const auto borders = computeBorders(data);

    //auto clf = trainFern<4, Output>(borders, data, targets, 50);
    auto clf = trainCascade<15, Output>(borders, data, targets, 10, 100, 0.5);
    Eigen::Matrix<Output, -1, 1> predicted = clf.predictMass(data);
    //std::cout << "predicted" << std::endl;
    double err = 0;
    for(int i = 0; i < predicted.rows(); ++i){
        err += (predicted[i] - targets[i]).norm();
        //std::cout << predicted[i] << " " << targets[i] << std::endl;
    }
    std::cout << "Train err " << err/nPoints * 100 << "% " << err << std::endl;

    Eigen::MatrixXf testData;
    Eigen::Matrix<Output, -1, 1> testTargets;
    const int nTestPoints = 10000;
    genCircleData(nTestPoints, testData, testTargets);

    err = 0;
    predicted = clf.predictMass(testData);
    for(int i = 0; i < predicted.rows(); ++i){
        err += (predicted[i] - testTargets[i]).norm();
        //std::cout << predicted[i] << " " << testTargets[i] << " " << err << std::endl;
    }
    std::cout << "Test err " << err/nTestPoints * 100 << "% " << err << std::endl;

    return 0;
}

