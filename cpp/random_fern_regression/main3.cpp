#include <iostream>

#include "Eigen/Core"
#include "fern.h"
#include "ferntrain.h"
#include "boosting.h"
#include "eigexport.h"

using namespace RandomFern;

typedef Eigen::Matrix<float, 1, 1> Output;

Output toEig(const float f)
{
    Eigen::Matrix<float, 1, 1> res;
    res[0] = f;
    return res;
}

Eigen::Matrix<Output, -1, 1> toEig(const Eigen::MatrixXf &vec)
{
    assert(vec.cols() == 1);
    Eigen::Matrix<Output, -1, 1> res(vec.rows());

    for(int i = 0; i < res.rows(); ++i)
        res(i, 0) = toEig(vec(i, 0));

    return res;
}

Eigen::Matrix<std::pair<float, float>, -1, 1> computeBorders(const Eigen::MatrixXf &data)
{
    Eigen::Matrix<std::pair<float, float>, -1, 1> res(data.cols());
    const Eigen::VectorXf maxs = data.array().colwise().maxCoeff();
    const Eigen::VectorXf mins = data.array().colwise().minCoeff();
    for(int i = 0; i < data.cols(); ++i)
        res(i, 0) = std::make_pair(mins[i], maxs[i]);
    
    return res;
}

inline Output l2Grad(const Output &residual)
{
    return residual;
}

inline Output l1Grad(const Output &residual)
{
    return toEig(fabs(residual[0])/residual[0]);
}

inline Output huberGrad(const Output &residual)
{
    const float delta = 2.0;
    const float absVal = fabs(residual[0]);
    if(absVal < delta)
        return residual;
    return toEig(delta * absVal / residual[0]);
}

int main()
{
    srand(42);
    
/*    const std::string dirPath    =           "/home/daiver/coding/jff/py/sk_linear_1/";*/
    //const std::string xTrainPath = dirPath + "X_train.txt";
    //const std::string yTrainPath = dirPath + "y_train.txt";
    //const std::string xTestPath  = dirPath + "X_test.txt";
    //const std::string yTestPath  = dirPath + "y_test.txt";

    const std::string dirPath    =           "/home/daiver/coding/jff/py/sk_grad_boost_1/";
    const std::string xTrainPath = dirPath + "X_train.txt";
    const std::string yTrainPath = dirPath + "y_train.txt";
    const std::string xTestPath  = dirPath + "X_test.txt";
    const std::string yTestPath  = dirPath + "y_test.txt";

    const Eigen::MatrixXf trainData        = EigRoutine::readMatFromTxt(xTrainPath);
    const Eigen::MatrixXf testData         = EigRoutine::readMatFromTxt(xTestPath);
    const Eigen::MatrixXf trainTargetsPure = EigRoutine::readMatFromTxt(yTrainPath);
    const Eigen::MatrixXf testTargetsPure  = EigRoutine::readMatFromTxt(yTestPath);
    const Eigen::Matrix<Output, -1, 1> trainTargets = toEig(trainTargetsPure);
    const Eigen::Matrix<Output, -1, 1> testTargets  = toEig(testTargetsPure);

    const auto borders = computeBorders(trainData);

    //best
    //Train err 1.65193
    //Test err 3.78963
    //auto clf = trainCascade<8, Output, l2Grad>(borders, trainData, trainTargets, 5, 1000, 0.02);
    //Train err 3.3061
    //Test err 3.66324
    //auto clf = trainCascade<5, Output, l1Grad>(borders, trainData, trainTargets, 5, 2000, 0.1);
    //auto clf = trainCascade<5, Output, huberGrad>(borders, trainData, trainTargets, 5, 2000, 0.1);
    //Train err 2.74637
    //Test err 3.3886
    //delta = 2.0
    //auto clf = trainCascade<8, Output, huberGrad>(borders, trainData, trainTargets, 5, 500, 0.1);
    auto clf = trainCascade<8, Output, huberGrad>(borders, trainData, trainTargets, 5, 500, 0.1);

    Eigen::Matrix<Output, -1, 1> predicted = clf.predictMass(trainData);
    double err = 0;
    for(int i = 0; i < predicted.rows(); ++i){
        err += (predicted[i] - trainTargets[i]).squaredNorm();
        //std::cout << predicted[i] << " " << trainTargets[i] << std::endl;
    }
    std::cout << "Train err " << err/trainData.rows() << std::endl;

    predicted = clf.predictMass(testData);
    err = 0;
    for(int i = 0; i < predicted.rows(); ++i){
        err += (predicted[i] - testTargets[i]).squaredNorm();
        //std::cout << predicted[i] << " " << testTargets[i] << std::endl;
    }
    std::cout << "Test err " << err/testData.rows() << std::endl;

    return 0;
}

