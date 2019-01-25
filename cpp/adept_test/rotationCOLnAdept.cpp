#include <stdio.h>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <adept.h>

#include "cppoptlib/solver/isolver.h"
#include "cppoptlib/solver/gradientdescentsolver.h"
#include "cppoptlib/solver/conjugatedgradientdescentsolver.h"
#include "cppoptlib/solver/bfgssolver.h"
#include "cppoptlib/solver/neldermeadsolver.h"
#include "cppoptlib/solver/newtondescentsolver.h"
#include "cppoptlib/solver/lbfgssolver.h"
#include "cppoptlib/solver/cmaessolver.h"

template<class Scalar, class Derived>
Eigen::Matrix<Scalar, 3, 1> func(
        const Eigen::Quaternion<Scalar> &rotation,
        const Eigen::Matrix<Scalar, 3, 1> &translation,
        const Eigen::MatrixBase<Derived> &point)
{
    //return point + translation;

    //Eigen::Matrix<Scalar, 3, 3> rotMat = Eigen::Quaternion<Scalar>(
                            //rotation.w()*sum, rotation.x()*sum, rotation.y()*sum, rotation.z()*sum
                            //).matrix();
    //std::cout << "tmp " << std::endl << (rotMat ) << std::endl;
    //return rotMat * point + translation;
    return rotation.matrix() * point + translation;
    //return point + translation;
}

template<class Scalar>
Scalar costFunc(
        const Eigen::Quaternion<Scalar> &rotation,
        const Eigen::Matrix<Scalar, 3, 1> &translation,
        const Eigen::Matrix<Scalar, -1,  3> &x,
        const Eigen::Matrix<Scalar, -1,  3> &y)
{
    Scalar err = 0;
    const Scalar &sum = 1.0/sqrt(rotation.w() * rotation.w() + 
                          rotation.x() * rotation.x() + 
                          rotation.y() * rotation.y() + 
                          rotation.z() * rotation.z());
    Eigen::Quaternion<Scalar> rot2(
                            rotation.w()*sum, rotation.x()*sum, rotation.y()*sum, rotation.z()*sum);
    for(int row = 0; row < x.rows(); ++row){
        Eigen::Matrix<Scalar, 3, 1> pointX = x.template block<1, 3>(row, 0).transpose();
        Eigen::Matrix<Scalar, 3, 1> pointY = y.template block<1, 3>(row, 0).transpose();
        Eigen::Matrix<Scalar, 3, 1> pointR = func(rot2, translation, pointX);
        //Eigen::Matrix<Scalar, 3, 1> pointR = func(rotation, translation, pointX);
        Eigen::Matrix<Scalar, 3, 1> tmp = pointR - pointY;
        err += tmp.dot(tmp);
/*        std::cout << row << " " << err << " " << std::endl*/
                  //<< "\t" << pointX.transpose()
                  //<< "|" << pointY.transpose()
                  //<< "|" << pointR.transpose()
                  //<< "|" << rotation.w() << " " << rotation.x() << " " << rotation.y() << " " << rotation.z()
                  /*<< "|" << tmp.transpose() << std::endl;*/
    }
    return err;// + beta.dot(beta);
}

template<class Scalar>
Scalar costFunc(
        const Eigen::Matrix<Scalar, -1, 1> &vars,
        const Eigen::Matrix<Scalar, -1, 3> &X,
        const Eigen::Matrix<Scalar, -1, 3> &y)
{
    Eigen::Quaternion<Scalar> rotation(vars[0], vars[1], vars[2], vars[3]);
    //std::cout << "rotation " << rotation.w() << " " << rotation.x() << " " << rotation.y() << " " << rotation.z() << std::endl;
    //std::cout << "rotation " << vars << std::endl;
    Eigen::Matrix<Scalar, 3, 1> translation = vars.template block<3, 1>(4, 0);
    return costFunc<Scalar>(rotation, translation, X, y);
}

using adept::adouble;

template<class T>
class RotationProblem : public cppoptlib::Problem<T>
{
public:
    RotationProblem(
        const Eigen::Matrix<T, -1, 3> &X,
        const Eigen::Matrix<T, -1, 3> &y){
        this->X = X;
        this->y = y;
        XAD = X.template cast<adouble>();
        yAD = y.template cast<adouble>();
    }

    T value(const cppoptlib::Vector<T> &vars)
    {
        Eigen::Quaternion<T> rotation(vars.template block<4, 1>(0, 0));
        Eigen::Matrix<T, 3, 1> translation = vars.template block<3, 1>(4, 0);
        return costFunc(rotation, translation, X, y);
    }

    void gradient(const cppoptlib::Vector<T> &vars, cppoptlib::Vector<T> &grad)
    {
        cppoptlib::Vector<adouble> varsAD = vars.template cast<adouble>();
        //Eigen::Quaternion<adouble> rotation(vars.template block<4, 1>(0, 0));
        //Eigen::Matrix<adouble, 3, 1> translation = vars.template block<3, 1>(4, 0);
        stack.new_recording();
        adouble funcRes = costFunc<adouble>(varsAD, XAD, yAD);
        funcRes.set_gradient(1.0);
        stack.compute_adjoint();
        double err = funcRes.value();
        //printf("err = %f\n", err);
        //for(int i = 0; i < 4; ++i)
        for(int i = 0; i < grad.rows(); ++i)
            grad[i] = varsAD[i].get_gradient();
        //std::cout << "Err " << costFunc(vars, X, y) << std::endl;
        //std::cout << grad.transpose() << std::endl;
        //std::cout << "vars " << vars.transpose() << std::endl;
    }

protected:
    adept::Stack stack;
    Eigen::Matrix<T, -1, 3> X;
    Eigen::Matrix<T, -1, 3> y;
    Eigen::Matrix<adouble, -1, 3> XAD;
    Eigen::Matrix<adouble, -1, 3> yAD;
};

int main()
{
    //Eigen::Quaternion<adouble> rotation;
/*    Eigen::Quaternion<double> rotation(Eigen::AngleAxisd(3, Eigen::Vector3d::UnitY()));*/
    //Eigen::Vector3d translation;
    //translation << 0, 10, 0;
    //Eigen::Vector3d point;
    //point << 1, 0, 0;
    /*std::cout << func(rotation, translation, point) << std::endl;*/


    Eigen::Matrix<double, -1, 3> X(9, 3);
    X << 1, 0, 0,
         0, 1, 0,
         0, 0, 1,
         1, 1, 0,
         0, 1, 1,
         2, 0.3, 1,
         0, -2.2, -1,
         9, 1, 2,
         1, 2, 3
             ;

    Eigen::Matrix<double, -1, 3> y(X.rows(), 3);
    //y << -1, 0, 0,
          //0, 1, 0,
          //1, 0, 0;

    Eigen::Quaterniond realRotation(Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitY()));
    Eigen::Vector3d realTranslate;
    realTranslate << 3, 0, 0;

    for(int row = 0; row < X.rows(); ++row)
        y.row(row) = func(realRotation, realTranslate, X.row(row).transpose()).transpose();

    std::cout << "Y" << std::endl << y << std::endl;

    Eigen::VectorXd vars(7);
    vars << 1, 0, 0, 0, 0, 0, 0;
    printf("err = %f\n", costFunc<double>(vars, X, y));
    //std::cout << Eigen::Quaterniond(1, 0, 0, 0) * X.block<1, 3>(1, 0).transpose() << std::endl;
    //std::cout << " " << Eigen::Quaterniond(1, 0, 0, 0).w();
    RotationProblem<double> problem(X, y);
    std::cout << "Check " << problem.checkGradient(vars) << std::endl;
    //cppoptlib::GradientDescentSolver<double> solver;
    //cppoptlib::ConjugatedGradientDescentSolver<double> solver;
    cppoptlib::BfgsSolver<double> solver;
    ////cppoptlib::NelderMeadSolver<double> solver;
    ////cppoptlib::NewtonDescentSolver<double> solver;
    ////cppoptlib::LbfgsSolver<double> solver;
    ////cppoptlib::CMAesSolver<double> solver;
    solver.minimize(problem, vars);
    std::cout << vars.transpose() << std::endl;
    printf("err = %f\n", costFunc(vars, X, y));
    return 0;
}
