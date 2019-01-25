#include <iostream>
#include <stdlib.h>

#define VIENNACL_WITH_EIGEN 1
//#define VIENNACL_WITH_OPENCL

#include "Eigen/Core"
#include "Eigen/Sparse"
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/coordinate_matrix.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/norm_1.hpp"
#include "viennacl/linalg/norm_inf.hpp"
#include "viennacl/linalg/ilu.hpp"
#include "viennacl/linalg/jacobi_precond.hpp"
#include "viennacl/linalg/cg.hpp"
#include "viennacl/linalg/bicgstab.hpp"
#include "viennacl/linalg/gmres.hpp"
#include "viennacl/io/matrix_market.hpp"


class GaussNewtonSolver
{
public:
    GaussNewtonSolver()
    {
        viennacl::linalg::cg_tag my_cg_tag(1e-5, 20);
        my_cg_solver = new viennacl::linalg::cg_solver<viennacl::vector<float>>(my_cg_tag);
    }

    ~GaussNewtonSolver()
    {
        delete my_cg_solver;
    }

    void gaussNewtonStep(
            const Eigen::VectorXf &vars,
            const Eigen::SparseMatrix<float> &jacobian,
            const Eigen::VectorXf &residuals,
            Eigen::VectorXf &result)
    {
        jacT = jacobian.transpose();
        hessian = jacT * jacobian;
        rhs = -jacT * residuals;
        viennacl::copy(hessian, A);
        viennacl::copy(rhs, b);
        viennacl::vector<float> res = (*my_cg_solver)(A, b);
        viennacl::copy(res, result);
    }

private:
    Eigen::VectorXf rhs;
    Eigen::SparseMatrix<float> jacT;
    Eigen::SparseMatrix<float> hessian;
    viennacl::compressed_matrix<float> A;
    viennacl::vector<float> b;
    viennacl::linalg::cg_solver<viennacl::vector<float> > *my_cg_solver;
};

void test03()
{
    
}

int main()
{
    test03();
    return 0;
}
