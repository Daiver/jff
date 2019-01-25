#ifndef FUNCTION_H
#define FUNCTION_H

#include <QtGlobal>
#include <vector>
#include "enums.h"
#include "expressioninfo.h"

namespace DCP {

class Function
{
public:
    static Monotonicity monotonicity(const int position, const ExpressionInfo &expr)
    {
        Q_UNUSED(position);
        Q_UNUSED(expr);
        return Monotonicity::Unknown;
    }

    static Sign sign(const std::vector<ExpressionInfo> &exprs)
    {
        Q_UNUSED(exprs);
        return Sign::Unknown;
    }

    static Curvature curvature(const std::vector<ExpressionInfo> &exprs)
    {
        Q_UNUSED(exprs);
        return Curvature::Unknown;
    }
};

}

#endif // FUNCTION_H
