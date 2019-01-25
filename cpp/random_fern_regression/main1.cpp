#include <iostream>

#include "Eigen/Core"
#include "fern.h"
#include "ferntrain.h"

using namespace RandomFern;

Eigen::Matrix<float, 1, 1> toEig(const float f)
{
    Eigen::Matrix<float, 1, 1> res;
    res[0] = f;
    return res;
}

int main()
{
    srand(42);
    Eigen::MatrixXf data(4, 2);
    data << 0, 0,
            1, 0,
            0, 1,
            1, 1;
    Eigen::Matrix<std::pair<float, float>, -1, 1> borders(data.cols());
    borders << std::make_pair(0, 1),
               std::make_pair(0, 1);
    typedef Eigen::Matrix<float, 1, 1> Output;
    Eigen::Matrix<Output, -1, 1> targets(data.rows(), 1);
    targets << toEig(0), toEig(1), toEig(1), toEig(0);

    auto fern = trainFern<2, Output>(borders, data, targets, 5);
    std::cout << "feat inds" << std::endl;
    std::cout << fern.featInds << std::endl;
    std::cout << "thresholds" << std::endl;
    std::cout << fern.thresholds << std::endl;
    const Eigen::Matrix<Output, -1, 1> predicted = fern.predict(data);
    std::cout << "predicted" << std::endl;
    for(int i = 0; i < predicted.rows(); ++i){
        std::cout << predicted[i] << std::endl;
    }
    //std::cout << sizeof(fern.mBins);
    return 0;
}
