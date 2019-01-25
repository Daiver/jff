#include <stdio.h>
#include <adept.h>
#include <math.h>
#include <vector>

using adept::adouble;

adouble func(const adouble &a, const adouble &b, const adouble &x)
{
    return a * x + b;
}

void gradFunc(
        const double &a, 
        const double &b, 
        const std::vector<double> &xs,
        const std::vector<double> &ys,
        double &da, double &db)
{
    da = 0;
    db = 0;
    for(int i = 0; i < xs.size(); ++i){
        da += 2*(a*xs[i] + b - ys[i])*xs[i];
        db += 2*(a*xs[i] + b - ys[i]);
    }
}

adouble costFunc(
        const adouble &a, 
        const adouble &b, 
        const std::vector<double> &xs,
        const std::vector<double> &ys)
{
    adouble res = 0;
    for(int i = 0; i < xs.size(); ++i)
        res += pow(func(a, b, xs[i]) - ys[i], 2);
    return res;
}

void optimize(
        const std::vector<double> &xs,
        const std::vector<double> &ys,
        const std::pair<double, double> &initVars,
        const float step,
        const int nIter)
{
    adept::Stack stack;
    adouble vars[2] = {initVars.first, initVars.second};


    for(int iter = 0; iter < nIter; ++iter){
        stack.new_recording();
        adouble funcRes = costFunc(vars[0], vars[1], xs, ys);
        funcRes.set_gradient(1.0);
        stack.compute_adjoint();
        double err = funcRes.value();
        double grad[2] = {vars[0].get_gradient(), vars[1].get_gradient()};
        //double grad[2];
        //gradFunc(vars[0].value(), vars[1].value(), xs, ys, grad[0], grad[1]);

        printf("err=%f da=%f db=%f a=%f b=%f\n", 
                err, grad[0], grad[1],
                vars[0].value(), vars[1].value());
        vars[0] -= step*grad[0];
        vars[1] -= step*grad[1];
    }

}

int main()
{
    std::vector<double> xs = {
        1.0, 2.0, 3.0, 4.0
    };
    std::vector<double> ys = {
        2.0, 3.0, 4.0, 5.0
    };
    optimize(xs, ys, std::make_pair(4, -4), 0.005, 500);
    return 0;
}
