#include <stdio.h>
#include <iostream>

#include <Eigen/Core>
#include <adept.h>

#include "cppoptlib/solver/isolver.h"
#include "cppoptlib/solver/gradientdescentsolver.h"
#include "cppoptlib/solver/conjugatedgradientdescentsolver.h"
#include "cppoptlib/solver/bfgssolver.h"

template<class Scalar>
Scalar func(
        const Eigen::Matrix<Scalar, -1, 1> &beta,
        const Eigen::Matrix<Scalar, -1, 1> &x)
{
    return beta.dot(x);
}

template<class Scalar>
Scalar costFunc(
        const Eigen::Matrix<Scalar, -1,  1> &beta,
        const Eigen::Matrix<Scalar, -1, -1> &X,
        const Eigen::Matrix<Scalar, -1,  1> &y)
{
    Eigen::Matrix<Scalar, -1, 1> tmp = (X * beta - y);
    return tmp.dot(tmp);// + beta.dot(beta);
}

using adept::adouble;

template<class T>
class LinearRegression : public cppoptlib::Problem<T>
{
public:
    LinearRegression(
        const Eigen::Matrix<T, -1, -1> &X,
        const Eigen::Matrix<T, -1,  1> &y){
        this->X = X;
        this->y = y;
        XAD = X.template cast<adouble>();
        yAD = y.template cast<adouble>();
    }

    T value(const cppoptlib::Vector<T> &vars)
    {
        return costFunc(vars, X, y);
    }

    void gradient(const cppoptlib::Vector<T> &vars, cppoptlib::Vector<T> &grad)
    {
        cppoptlib::Vector<adouble> varsAD = vars.template cast<adouble>();
        stack.new_recording();
        adouble funcRes = costFunc<adouble>(varsAD, XAD, yAD);
        funcRes.set_gradient(1.0);
        stack.compute_adjoint();
        double err = funcRes.value();
        //printf("err = %f\n", err);
        for(int i = 0; i < grad.rows(); ++i)
            grad[i] = varsAD[i].get_gradient();
    }

protected:
    adept::Stack stack;
    Eigen::Matrix<T, -1, -1> X;
    Eigen::Matrix<T, -1,  1> y;
    Eigen::Matrix<adouble, -1, -1> XAD;
    Eigen::Matrix<adouble, -1,  1> yAD;
};

int main()
{
    Eigen::MatrixXd X(4, 2);
    X << 1, 1,
         2, 1,
         3, 1,
         4, 1;
    Eigen::VectorXd y(4);
    y << 3, 5, 7, 9;
    Eigen::VectorXd beta(2);
    //beta << 2, 1;
    beta << 4, -4;
    printf("err = %f\n", costFunc(beta, X, y));
    LinearRegression<double> problem(X, y);
    cppoptlib::GradientDescentSolver<double> solver;
    //cppoptlib::ConjugatedGradientDescentSolver<double> solver;
    //cppoptlib::BfgsSolver<double> solver;
    solver.minimize(problem, beta);
    std::cout << beta << std::endl;
    printf("err = %f\n", costFunc(beta, X, y));
    //optimize(beta, X, y, 0.005, 500);
    return 0;
}
