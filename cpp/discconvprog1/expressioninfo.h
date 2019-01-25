#ifndef EXPRESSIONINFO_H
#define EXPRESSIONINFO_H

#include "enums.h"

namespace DCP {

class ExpressionInfo
{
public:
    Sign sign;
    Curvature curvature;
};

std::string to_string(const ExpressionInfo &info);
}

#endif // EXPRESSIONINFO_H
