#include "eigen3/Eigen/Core"
#include <stdio.h>
#include <iostream>

void foo(const Eigen::Ref<Eigen::MatrixXf> &mat)
{
    std::cout << mat << std::endl;
}

int main()
{
    Eigen::MatrixXf m1(6, 1);
    m1 << 1, 2, 3, 4, 5, 6;
    foo(m1);
    Eigen::Map<Eigen::MatrixXf> m2(m1.data() + 2, 2, 2);
    Eigen::Map<Eigen::MatrixXf> m3(m1.data(), 2, 2);
    foo(m2);
    foo(m3);
    return 0;
}
