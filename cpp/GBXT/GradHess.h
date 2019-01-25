#ifndef GRADHESS_H
#define GRADHESS_H

#include <Eigen/Core>

template<class Scalar>
class GradHess
{
public:
    GradHess(): grad(0), hess(0)
    {   }

    GradHess(const Scalar &grad): grad(grad), hess(0)
    {   }

    GradHess(const Scalar grad, const Scalar hess): grad(grad), hess(hess)
    {   }

    Scalar grad;
    Scalar hess;


    GradHess<Scalar> &operator += (const GradHess<Scalar> &other)
    {
        this->grad += other.grad;
        this->hess += other.hess;
        return *this;
    }

    GradHess<Scalar> &operator -= (const GradHess<Scalar> &other)
    {
        this->grad -= other.grad;
        this->hess -= other.hess;
        return *this;
    }
};

template<
    class Scalar,
    class ValueScalar,
    class LossGrad, 
    class LossHessian
    >
Eigen::Matrix<GradHess<Scalar>, -1, 1> computeGradAndHessPerSample(
        const Eigen::Matrix<ValueScalar, -1, 1> &targetValues,
        const Eigen::Matrix<ValueScalar, -1, 1> &answersFromPreviousStage,
        LossGrad     &lossGrad,
        LossHessian  &lossHessian)
{
    const int nSamples = targetValues.rows();
    Eigen::Matrix<GradHess<Scalar>, -1, 1> gradAndHessPerSample(nSamples);
    for(int sampleInd = 0; sampleInd < nSamples; ++sampleInd){
        gradAndHessPerSample[sampleInd] = GradHess<Scalar>(
                lossGrad(targetValues[sampleInd], answersFromPreviousStage[sampleInd]),
                lossHessian(targetValues[sampleInd], answersFromPreviousStage[sampleInd]));
    }
    return gradAndHessPerSample;
}

#endif
