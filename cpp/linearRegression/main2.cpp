#include <iostream>
#include <Eigen/Dense>
#include <cpplinq.hpp>

template<class Derived, class Derived2, class Scalar>
float residualWithoutKWeight(
        const Eigen::MatrixBase<Derived> &weights,
        const Eigen::MatrixBase<Derived2> &sample,
        const Scalar output,
        const int k)
{
    return weights.dot(sample) - output - weights(k, 0)*sample(k, 0);
}

float rss(
        const Eigen::VectorXf &weights, 
        const Eigen::MatrixXf &data,
        const Eigen::VectorXf &output)
{
    auto errors = data * weights - output;
    return (errors.transpose() * errors).sum();
}

void coordinateDescentRidge(
        const Eigen::MatrixXf &data,
        const Eigen::VectorXf &output,
        Eigen::VectorXf &weights, 
        const float lambda,
        const int nIters)
{
    const int nExamples = data.rows();
    const int nFeatures = data.cols();
    for(int iter = 0; iter < nIters; ++iter){
        const int featureInd = iter % nFeatures;
        float rho = 0;
        for(int i = 0; i < nExamples; ++i)
            rho += residualWithoutKWeight(
                    weights,
                    data.row(i).transpose(),
                    output[i],
                    featureInd) * data(i, featureInd);
        auto column = data.col(featureInd);
        float sumOfColumn = (column.transpose() * column).sum();
        weights[featureInd] = -rho/(sumOfColumn + lambda);
        const float err = rss(weights, data, output);
        std::cout << "iter " << iter << " err " << err << std::endl;
        std::cout << weights << std::endl;
    }
}

float coordinateDescentStepLasso(
        const float weight,
        const float sumOfColumn,
        const float rho,
        const float lambda)
{
    if(-rho + lambda/2.0 < 0)
        return (-rho + lambda/2.0)/sumOfColumn;
    if(-rho - lambda/2.0 > 0)
        return (-rho - lambda/2.0)/sumOfColumn;
    return 0;
}

void coordinateDescentLasso(
        const Eigen::MatrixXf &data,
        const Eigen::VectorXf &output,
        Eigen::VectorXf &weights, 
        const float lambda,
        const int nIters,
        const bool verbose)
{
    const int nExamples = data.rows();
    const int nFeatures = data.cols();
    for(int iter = 0; iter < nIters; ++iter){
        const int featureInd = iter % nFeatures;
        float rho = 0;
        for(int i = 0; i < nExamples; ++i)
            rho += residualWithoutKWeight(
                    weights,
                    data.row(i).transpose(),
                    output[i],
                    featureInd) * data(i, featureInd);
        auto column = data.col(featureInd);
        float sumOfColumn = (column.transpose() * column).sum();
        weights[featureInd] = coordinateDescentStepLasso(weights[featureInd], sumOfColumn, rho, lambda);
        if(verbose){
            const float err = rss(weights, data, output);
            std::cout << "iter " << iter << " err " << err << std::endl;
            std::cout << weights << std::endl;
        }
    }
}

int main()
{
    Eigen::MatrixXf data(5, 2);
    data << 1, 1,
            2, 1,
            3, 1,
            4, 1,
            5, 1;
    Eigen::VectorXf output(data.rows());
    output << 3 , 5 , 7 , 9 , 11;

    Eigen::VectorXf weights(data.cols());
    weights << 3, 1;

    coordinateDescentLasso(data, output, weights, 10.1, 10, true);
    //coordinateDescentRidge(data, output, weights, 0.0, 100);
    //std::cout << residualWithoutKWeightRidge(weights, data.row(0).transpose(), output[0], 0) << std::endl;
    return 0;
}
