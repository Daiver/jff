#include <iostream>
#include <stdlib.h>

//#define VIENNACL_WITH_OPENCL

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

void test01();
void test02();


int main()
{
    test02();
    return 0;
}

//float random()
//{
//    return ((float)rand()) / RAND_MAX;
//}

void test02()
{
    typedef float        ScalarType;
    viennacl::scalar<ScalarType> vcl_s1;
    viennacl::scalar<ScalarType> vcl_s2 = 1.0;
    viennacl::scalar<ScalarType> vcl_s3 = 1.0;
    ScalarType s1 = 3.1415926;
    ScalarType s2 = 2.71763;
    ScalarType s3 = 42.0;
    //typedef double    ScalarType; //use this if your GPU supports double precision
     
    // Define a few CPU vectors using the STL
    const int size = 10000;  
    std::vector<ScalarType>      std_vec1(size);
    std::vector<ScalarType>      std_vec2(size);
    std::vector<ScalarType>      std_vec3(size);
     
    // Define a few GPU vectors using ViennaCL
    viennacl::vector<ScalarType> vcl_vec1(size);
    viennacl::vector<ScalarType> vcl_vec2(size);
    viennacl::vector<ScalarType> vcl_vec3(size);
     
    // Fill the CPU vectors with random values:
    // (random<> is a helper function defined elsewhere)
    

    for (unsigned int i = 0; i < size; ++i)
    {
      std_vec1[i] = random(); 
      //vcl_vec2[i] = random();  //also works for GPU vectors, but is slow!
      std_vec3[i] = random(); 
    }
      
    // Copy the CPU vectors to the GPU vectors and vice versa
    copy(std_vec1.begin(), std_vec1.end(), vcl_vec1.begin()); //either the STL way
    copy(vcl_vec2.begin(), vcl_vec2.end(), std_vec2.begin());
    copy(std_vec3, vcl_vec3); //or using the short hand notation
    copy(vcl_vec2, std_vec2);
     
    // Compute the inner product of two GPU vectors and write the result to either CPU or GPU
    vcl_s1 = viennacl::linalg::inner_prod(vcl_vec1, vcl_vec2);
    s1     = viennacl::linalg::inner_prod(vcl_vec1, vcl_vec2);
     
    // Compute norms:
    vcl_s2 = viennacl::linalg::norm_2(vcl_vec2);
    s3     = viennacl::linalg::norm_inf(vcl_vec3);
      
    // Use viennacl::vector via the overloaded operators just as you would write it on paper:
    for(int i = 0; i < 30000; ++i){
        vcl_vec1 = vcl_s1 * vcl_vec2 / vcl_s3;
        s1     = viennacl::linalg::norm_1(vcl_vec1);
    }
    vcl_vec1 = vcl_vec2 / vcl_s1 + vcl_s2 * (vcl_vec1 - vcl_s2 * vcl_vec2);    
    std::cout << "HI" << std::endl;
}

void test01()
{
    std::cout << "HI" << std::endl;
    typedef float        ScalarType;
    ScalarType s1 = 3.1415926;
    ScalarType s2 = 2.71763;
    ScalarType s3 = 42.0;

    viennacl::scalar<ScalarType> vcl_s1;
    viennacl::scalar<ScalarType> vcl_s2 = 1.0;
    viennacl::scalar<ScalarType> vcl_s3 = 1.0;
    
    vcl_s1 = s1;
    s2 = vcl_s2;
    vcl_s3 = s3;

    s1 += s2;
    vcl_s1 += vcl_s2;
     
    s1 = s2 + s3;
    vcl_s1 = vcl_s2 + vcl_s3;
      
    s1 = s2 + s3 * s2 - s3 / s1;
    vcl_s1 = vcl_s2 + vcl_s3 * vcl_s2 - vcl_s3 / vcl_s1;
      
    // Operations can also be mixed:
    vcl_s1 = s1 * vcl_s2 + s3 - vcl_s3;
       
    // Output stream is overloaded as well:
    std::cout << "CPU scalar s2: " << s2 << std::endl;
    std::cout << "GPU scalar vcl_s2: " << vcl_s2 << std::endl;
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
