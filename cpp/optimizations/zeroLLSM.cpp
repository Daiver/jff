#include <iostream>

#include <eigen3/Eigen/Dense>

int main()
{
    Eigen::MatrixXf testA = Eigen::MatrixXf::Zero(3, 2);
    testA << 0, 0,
             3, 4,
             0, 0;

    Eigen::VectorXf testB = Eigen::VectorXf::Zero(3);
    testB << 0, 2, 0;
    
    Eigen::VectorXf solution = (testA.transpose() * testA).llt().solve(testA.transpose() * testB);

    std::cout << solution << std::endl;

    return 0;
}
