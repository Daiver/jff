#include <iostream>
#include "eigen3/Eigen/Dense"

using namespace Eigen;

int main()
{
    Matrix<float, 2, 3> m1, m2;
    m1 << 1, 2, 3, 4, 5, 6;
    m2 << 7, 8, 9, 10, 11, 12;
    std::cout << m1 + m2;
}
/*    MatrixXd m(2,2);*/
    //m(0,0) = 3;
    //m(1,0) = 2.5;
    //m(0,1) = -1;
    //m(1,1) = m(1,0) + m(0,1);
    /*std::cout << m << std::endl;*/

