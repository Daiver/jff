#include <stdio.h>
#include <eigen3/Eigen/Core>
#include <adept.h>

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
    return tmp.dot(tmp);
}

void optimize(
        const Eigen::Matrix<double, -1,  1> &betaInitial,
        const Eigen::Matrix<double, -1, -1> &X,
        const Eigen::Matrix<double, -1,  1> &y,
        const float step,
        const int nIters)
{
    using adept::adouble;
    adept::Stack stack;
    Eigen::Matrix<adouble, -1, 1> beta = betaInitial.cast<adouble>();
    for(int iter = 0; iter < nIters; ++iter){
        stack.new_recording();
        adouble funcRes = costFunc<adouble>(beta, 
                X.cast<adouble>(), y.cast<adouble>());
        funcRes.set_gradient(1.0);
        stack.compute_adjoint();
        double err = funcRes.value();
        Eigen::VectorXd grad(beta.rows());
        for(int i = 0; i < grad.rows(); ++i)
            grad[i] = beta[i].get_gradient();
        beta -= step * grad.cast<adouble>();
        printf("err = %f a = %f b = %f\n", err, beta[0].value(), beta[1].value());
    }
}

int main()
{
    Eigen::MatrixXd X(4, 2);
    X << 1, 1,
         2, 1,
         3, 1,
         4, 1;
    Eigen::VectorXd y(4);
    y << 2, 3, 4, 5;
    Eigen::VectorXd beta(2);
    beta << 4, -4;
    printf("err = %f\n", costFunc(beta, X, y));
    optimize(beta, X, y, 0.005, 500);
    return 0;
}
