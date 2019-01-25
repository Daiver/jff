#include "IpIpoptApplication.hpp"

#include "myproblem1.h"
#include "myproblem2.h"

using namespace Ipopt;

int main()
{    
//    SmartPtr<TNLP> mynlp = new MyProblem1();
    SmartPtr<TNLP> mynlp = new MyProblem2();
    SmartPtr<IpoptApplication> app = new IpoptApplication();

    app->Initialize();
    app->Options()->SetStringValue("hessian_approximation", "limited-memory");
    ApplicationReturnStatus status = app->OptimizeTNLP(mynlp);

    return 0;
}
