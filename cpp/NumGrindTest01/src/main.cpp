#include <iostream>

#include "utils.h"
#include "numgrind.h"
#include "Eigen/Core"

float sigmoid(float z)
{
        return 1.0/(1.0 + exp(-z));
}

float sigmoidDer(float z)
{
        return sigmoid(z) * sigmoid(1.0 - z);
}

int main() {
    std::cout << "Hello, World!" << std::endl;

    //auto w1 = GNMatrixVariable(2, 2, {0, 1, 2, 3});
    auto w1 = GNMatrixVariable(2, 1, {0, 1});
    auto b1 = GNScalarVariable(2);
    Eigen::MatrixXf data(2, 4);
    Eigen::VectorXf targets(4);
    data << 
        0, 0,
        1, 0,
        0, 1,
        1, 1;

    targets << 0, 0, 0, 1;

    auto x = GNMatrixConstant(data);
    auto y = GNMatrixConstant(targets);

/*    auto n3 = GNVectorElementWiseProduct(&n1, &n2);*/
    //auto n4 = GNDotProduct(&n3, &n1);

    //auto graph = n4;

    //Eigen::VectorXf vars = utils::vec2EVecf({1, 2, 3, 4});
    //Eigen::VectorXf grad = Eigen::VectorXf::Zero(vars.size());
    //graph.forwardPass(vars);
    //graph.backwardPass(1, grad);

    //std::cout << graph.toString() << std::endl;
    //for(int i = 0; i < grad.size(); ++i)
        //std::cout << i << ":" << grad[i] << " ";
    //std::cout << std::endl;

    return 0;
}
