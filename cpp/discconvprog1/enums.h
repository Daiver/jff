#ifndef ENUMS_H
#define ENUMS_H

#include <string>

namespace DCP {

enum class Sign
{
    Unknown,
    Negative,
    Positive,
};

std::string to_string(const Sign signInfo);

enum class Curvature
{
    Constant,
    Affine,
    Convex,
    Concave,
    Unknown,

};

bool isConvexOrAffine(const Curvature curv);
bool isConcaveOrAffine(const Curvature curv);

std::string to_string(const Curvature curvInfo);

enum class Monotonicity
{
    Decreasing,
    Increasing,
    Unknown
};


}

#endif // ENUMS_H
