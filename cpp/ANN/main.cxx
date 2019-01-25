//#define NDEBUG

#include <stdio.h>
#include <iostream>

#define MATLAB
#include "cppoptlib/solver/conjugatedgradientdescentsolver.h"
#include "CommonEigenRoutine/commoneigenroutine.h"
#include "Layers/layer.h"
#include "ActFunctions/actfunctions.h"
#include "adept.h"

using adept::adouble;

template<typename T>
T costFunc(
        const ANN::Layer<T, ANN::linearFunc, ANN::linearDer> &layer,
        //const ANN::Layer<T, ANN::sigmoid, ANN::sigmoidDer> &layer,
        const ANN::Matrix<T> &weights, 
        const ANN::Matrix<T> &data,
        const ANN::Vector<T> &values)
{
    T res = 0;
    for(int i = 0; i < data.rows(); ++i){
        ANN::Vector<T> ans = layer.activate(weights, data.row(i));
        res += (ans[0] - values[i])*(ans[0] - values[i]);
    }
    return res;
}

template<typename T>
class RegressionProblem : public cppoptlib::Problem<T>
{
public:
    RegressionProblem(
        const Eigen::Matrix<T, -1, -1> &data,
        const Eigen::Matrix<T, -1,  1> &values){
        this->data = data;
        this->values = values;
        dataAD = data.template cast<adouble>();
        valuesAD = values.template cast<adouble>();
    }

    T value(const ANN::Vector<T> &vars)
    {
        ANN::Layer<T, ANN::linearFunc, ANN::linearDer> layer(1, 1);
        ANN::Matrix<T> weights(1, vars.rows());
        for(int i = 0; i < vars.rows(); ++i)
            weights(0, i) = vars[i];
        return costFunc<T>(
                layer, 
                CommonEigenRoutine::reshape(vars, vars.rows(), 1), 
                data, values);
    }
    
    adept::Stack stack;

    ANN::Matrix<T> data;
    ANN::Vector<T> values;
    ANN::Matrix<adouble> dataAD;
    ANN::Vector<adouble> valuesAD;
};

int main()
{
    //adept::Stack stack;
    ANN::Vector<double> weights(2);
    weights << 0.01, 0.01;
    ANN::Matrix<double> data(5, 1);
    data << 1, 2, 3, 4, 5;
    ANN::Vector<double> values(5);
    values << 0, 1, 2, 3, 4;
    RegressionProblem<double> problem(data, values);
    cppoptlib::ConjugatedGradientDescentSolver<double> solver;
    solver.minimize(problem, weights);
    std::cout << weights << std::endl;
    return 0;
}

int main1()
{
    std::cout << "Start" << std::endl;
    ANN::Layer<float, ANN::sigmoid, ANN::sigmoidDer> l(2, 1);

    Eigen::MatrixXf weights(1, 3);
    weights << 0.1, 0.2, 0.3;
    Eigen::VectorXf example(2);
    example << 2, 3;
    ANN::Layer<float, ANN::sigmoid, ANN::sigmoidDer>::ScalarType res = l.activate(weights, example)[0];
    std::cout << res << std::endl;
    //Eigen::MatrixXi v(4, 1);
    ////Eigen::VectorXi v(4);
    //v << 1, 2, 3, 4;
    //auto m = CommonEigenRoutine::reshape(v, 2, 2);
    ////auto m = Eigen::Map<Eigen::MatrixXi, Eigen::Aligned>(v.data(), 2, 2);
    ////auto m = Eigen::Map<Eigen::Matrix<int, -1, -1, Eigen::RowMajor>, Eigen::Aligned>(v.data(), 2, 2);
    //std::cout << v.transpose() << std::endl;
    /*std::cout << m << std::endl;*/
    return 0;
}
