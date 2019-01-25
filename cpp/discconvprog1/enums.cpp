#include "enums.h"

std::__cxx11::string DCP::to_string(const DCP::Sign signInfo)
{
    if(signInfo == Sign::Negative)
        return std::string("-");
    if(signInfo == Sign::Positive)
        return std::string("+");
    return std::string("+/-");
}

std::__cxx11::string DCP::to_string(const DCP::Curvature curvInfo)
{
    if(curvInfo == Curvature::Constant)
        return std::string("Const");
    if(curvInfo == Curvature::Affine)
        return std::string("Affine");
    if(curvInfo == Curvature::Convex)
        return std::string("Convex");
    if(curvInfo == Curvature::Concave)
        return std::string("Concave");
    return std::string("Unknown");
}

bool DCP::isConvexOrAffine(const DCP::Curvature curv)
{
    return (curv == Curvature::Affine || curv == Curvature::Convex || curv == Curvature::Constant);
}

bool DCP::isConcaveOrAffine(const DCP::Curvature curv)
{
    return (curv == Curvature::Affine || curv == Curvature::Concave || curv == Curvature::Constant);
}
