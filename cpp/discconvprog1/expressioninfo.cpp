#include "expressioninfo.h"


std::string DCP::to_string(const DCP::ExpressionInfo &info)
{
    return std::string("{Sign: ") + to_string(info.sign) + ", Curv: " + to_string(info.curvature) + "}";
}
