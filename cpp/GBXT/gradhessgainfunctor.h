#ifndef GRADHESSGAINFUNCTOR_H
#define GRADHESSGAINFUNCTOR_H

#include "GradHess.h"

template<class Scalar>
class GradHessGainFunctor
{
public:
    
    GradHessGainFunctor(const double lambda, const double gamma): lambda(lambda), gamma(gamma)
    {

    }

    Scalar operator() (
                const GradHess<Scalar> &gAll,
                const GradHess<Scalar> &gl,
                const GradHess<Scalar> &gr) const
    {
        Scalar allTerm = (gAll.grad * gAll.grad)/(gAll.hess*gAll.hess + lambda);
        Scalar lTerm   = (gl.grad * gl.grad)/(gl.hess * gl.hess + lambda);
        Scalar rTerm   = (gr.grad * gr.grad)/(gr.hess * gr.hess + lambda);
        return lTerm + rTerm - allTerm - gamma;
    }

    double lambda = 0;
    double gamma  = 0;

};

#endif
