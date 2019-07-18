#include <iostream>

#include "Eigen/Sparse"
#include "Eigen/PardisoSupport"

int main(int argc, char *argv[])
{    
    Eigen::SparseMatrix<float> mat(3, 3);
    mat.setIdentity();
    Eigen::PardisoLLT<Eigen::SparseMatrix<float>> solver;
    solver.analyzePattern(mat);
    std::cout << "solver.info(): " << solver.info() << std::endl;

    solver.analyzePattern(mat);
    std::cout << "solver.info(): " << solver.info() << std::endl;

    solver.analyzePattern(mat);
    std::cout << "solver.info(): " << solver.info() << std::endl;

    solver.analyzePattern(mat);
    std::cout << "solver.info(): " << solver.info() << std::endl;
    return 0;
}
