#ifndef ACTFUNCTIONS_H
#define ACTFUNCTIONS_H

#include <math.h>

namespace ANN {

template<typename Scalar>
Scalar sigmoid(const Scalar z);

template<typename Scalar>
Scalar sigmoidDer(const Scalar z);

template<typename Scalar>
Scalar linearFunc(const Scalar z);

template<typename Scalar>
Scalar linearDer(const Scalar z);

}




template<typename Scalar>
inline Scalar ANN::sigmoid(const Scalar z)
{
    return 1.0/(1.0 + exp(-z));
}

template<typename Scalar>
inline Scalar ANN::sigmoidDer(const Scalar z)
{
    return sigmoid(z) * sigmoid(1.0 - z);
}


template<typename Scalar>
inline Scalar ANN::linearFunc(const Scalar z)
{
    return z;
}

template<typename Scalar>
inline Scalar ANN::linearDer(const Scalar z)
{
    return 1.0;
}

#endif
