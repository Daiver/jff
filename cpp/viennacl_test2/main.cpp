#include <iostream>
#include <stdlib.h>

#define VIENNACL_HAVE_EIGEN
#define VIENNACL_WITH_OPENCL

#include "Eigen/Dense"
#include "Eigen/IterativeLinearSolvers"

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

#include <chrono>

template<typename TimeT = std::chrono::milliseconds>
struct measure
{
    template<typename F, typename ...Args>
    static typename TimeT::rep execution(F&& func, Args&&... args)
    {
        auto start = std::chrono::steady_clock::now();
        std::forward<decltype(func)>(func)(std::forward<Args>(args)...);
        auto duration = std::chrono::duration_cast< TimeT> 
                            (std::chrono::steady_clock::now() - start);
        return duration.count();
    }
};

int main()
{
    const int dim = 3000;
    Eigen::MatrixXf mat(dim, dim);
    Eigen::VectorXf vec(dim);
    Eigen::VectorXf solution(dim);
    mat.setRandom();
    //mat = mat.transpose() * mat;
    vec.setRandom();

    viennacl::matrix<float>                viennacl_densematrix(dim, dim);
    viennacl::matrix<float>                viennacl_densematrix2(dim, dim);
    viennacl::vector<float>                viennacl_vector(dim);
    viennacl::vector<float>                viennacl_solution(dim);


    auto f1 = [&]() {
        viennacl::copy(mat, viennacl_densematrix);
        viennacl::copy(vec, viennacl_vector);

        auto tmp = trans(viennacl_densematrix);
        viennacl_densematrix2 = 
            viennacl::linalg::prod(
                    tmp, viennacl_densematrix);
        viennacl_solution = solve(
                      viennacl_densematrix2,    
                      viennacl_vector,
                      viennacl::linalg::cg_tag());
        viennacl::copy(viennacl_solution, solution);
    };
    auto f2 = [&]() {
        //solution = mat.colPivHouseholderQr().solve(vec);
        mat = mat.transpose() * mat;
        Eigen::ConjugateGradient<Eigen::MatrixXf, Eigen::Lower | Eigen::Upper> cg;
        cg.compute(mat);
        solution = cg.solve(vec);
        /*using namespace Eigen;
        LeastSquaresConjugateGradient<Eigen::MatrixXf> lscg;
        lscg.compute(mat);
        solution = lscg.solve(vec);*/
    };

    float t1 = measure<>::execution(f1) / 1000.0;
    float t2 = measure<>::execution(f2) / 1000.0;
    std::cout << "GPU " << t1 << std::endl;
    std::cout << "CPU " << t2 << std::endl;
    std::cout << "CPU/GPU " << t2/t1 << std::endl;

    /*solution = solve(
                      mat,    
                      vec,
                      viennacl::linalg::cg_tag());*/
 
    /*std::cout << mat << std::endl << vec << std::endl;
    std::cout << "solution" << std::endl << solution << std::endl;
    std::cout << "real solution" << std::endl;
    std::cout << mat.colPivHouseholderQr().solve(vec) << std::endl;*/
    std::cout << "End" << std::endl;

    return 0;
}

/*void test02()
{
typedef float        ScalarType;
//typedef double    ScalarType; //use this if your GPU supports double precision
 
// Set up some ublas objects:
ublas::vector<ScalarType> ublas_rhs;
ublas::vector<ScalarType> ublas_result;
ublas::compressed_matrix<ScalarType> ublas_matrix;
 
// Set up some ViennaCL objects:
viennacl::vector<ScalarType> vcl_rhs;
viennacl::vector<ScalarType> vcl_result;
viennacl::compressed_matrix<ScalarType> vcl_matrix;
 
 
//
// Compute ILUT preconditioners for CPU and for GPU objects:
//
viennacl::linalg::ilut_tag ilut_conf(10, 1e-5);  //10 entries, rel. tol. 1e-5
typedef viennacl::linalg::ilut_precond< 
                    ublas::compressed_matrix<ScalarType> >     ublas_ilut_t;
//preconditioner for ublas objects:
ublas_ilut_t ublas_ilut(ublas_matrix, ilut_conf);   
 
viennacl::linalg::ilut_precond<
                    viennacl::compressed_matrix<ScalarType> >  vcl_ilut_t;
//preconditioner for ViennaCL objects:
vcl_ilut_t vcl_ilut(vcl_matrix, ilut_conf);
 
//
// Conjugate gradient solver without preconditioner:
//
ublas_result  = solve(ublas_matrix,   //using ublas objects on CPU
                      ublas_rhs,
                      viennacl::linalg::cg_tag());
vcl_result    = solve(vcl_matrix,     //using viennacl objects on GPU
                      vcl_rhs,
                      viennacl::linalg::cg_tag());
 
 
//
// Conjugate gradient solver using ILUT preconditioner
//
ublas_result  = solve(ublas_matrix,   //using ublas objects on CPU
                      ublas_rhs,
                      viennacl::linalg::cg_tag(), 
                      ublas_ilut);
vcl_result    = solve(vcl_matrix,     //using viennacl objects on GPU
                      vcl_rhs,
                      viennacl::linalg::cg_tag(),
                      vcl_ilut);
}*/
