#ifndef LAYER_H
#define LAYER_H

#include <iostream>

#include "Eigen/Core"
#include "../CommonEigenRoutine/commoneigenroutine.h"

namespace ANN {

template <typename T>
using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <typename T>
using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

template<typename Scalar, Scalar ActFunc(const Scalar), Scalar GradFunc(const Scalar)>
class Layer {
public:
    typedef Scalar ScalarType;
    //typedef Eigen::Ref<Matrix<Scalar>> MatrixR;
    //typedef Eigen::Ref<Vector<Scalar>> VectorR;

    Layer(const int nInputs, const int nOutputs): nInputs(nInputs), nOutputs(nOutputs) {}
    
    template<typename Derived1, typename Derived2>
    Vector<Scalar> activate(
            const Eigen::MatrixBase<Derived1> &weights, 
            const Eigen::MatrixBase<Derived2> &example) const;

protected:
    const int nInputs;
    const int nOutputs;
};

}





template<typename Scalar, Scalar ActFunc(const Scalar), Scalar GradFunc(const Scalar)>
template<typename Derived1, typename Derived2>
    ANN::Vector<Scalar> ANN::Layer<Scalar, ActFunc, GradFunc>::activate(
            const Eigen::MatrixBase<Derived1> &weights, 
            const Eigen::MatrixBase<Derived2> &example) const
{
    assert(example.rows() == nInputs);
    assert(weights.cols() == nInputs + 1);
    //std::cout << "In activate\n";
    ANN::Vector<Scalar> prod = weights.col(nInputs) 
                             + weights.block(0, 0, nOutputs, nInputs) * example;

    CommonEigenRoutine::mapMut(ActFunc, prod);
    
    return prod;
}

#endif
