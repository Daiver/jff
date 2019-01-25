#include<stdio.h>

#include <iosfwd>
#include <vector>
#include <cppad/cg.hpp>
#include <math.h>
//#include <cppad/cg/cg.hpp>

using namespace CppAD;
using namespace CppAD::cg;

typedef CG<double> CGD;
typedef AD<CGD> ADCG;



int main()
{

    /***************************************************************************
     *                               the model
     **************************************************************************/

    // independent variable vector
    CppAD::vector<ADCG> x(2);
    x[0] = 2.;
    x[1] = 3.;
    Independent(x);

    // dependent variable vector 
    CppAD::vector<ADCG> y(1);

    // the model
    ADCG a = x[1] + x[0] * x[0];
    y[0] = sin(a / 2);

    ADFun<CGD> fun(x, y); // the model tape

    /***************************************************************************
     *                        Generate the C source code
     **************************************************************************/

    /**
     * start the special steps for source code generation for a Jacobian
     */
    CodeHandler<double> handler;

    CppAD::vector<CGD> indVars(2);
    handler.makeVariables(indVars);

    CppAD::vector<CGD> jac = fun.Jacobian(indVars);
    //CppAD::vector<CGD> jac = fun.SparseJacobian(indVars);

    LanguageC<double> langC("double");
    LangCDefaultVariableNameGenerator<double> nameGen;

    std::ostringstream code;
    handler.generateCode(code, langC, jac, nameGen);
    std::cout << code.str();
    return 0;
}
