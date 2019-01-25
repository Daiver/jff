#include <iostream>

#include "expressioninfo.h"
#include "inference.h"
#include "functions.h"

namespace DCP {

ExpressionInfo variable()
{
    return ExpressionInfo{Sign::Unknown, Curvature::Affine};
}

ExpressionInfo constant()
{
    return ExpressionInfo{Sign::Unknown, Curvature::Constant};
}

ExpressionInfo operator+(const ExpressionInfo &a, const ExpressionInfo &b)
{
    return inference<Sum2>({a, b});
}

ExpressionInfo operator-(const ExpressionInfo &a)
{
    return inference<Minus>({a});
}

ExpressionInfo operator-(const ExpressionInfo &a, const ExpressionInfo &b)
{
    return inference<Minus2>({a, b});
}


ExpressionInfo abs(const ExpressionInfo &a)
{
    return inference<Abs>({a});
}

}

int main()
{
    using namespace DCP;
    auto expr = - abs(variable() + constant()) + variable();
    std::cout << to_string(expr) << std::endl;
    expr = abs(constant());
    std::cout << to_string(expr) << std::endl;
    expr = constant() + constant();
    std::cout << to_string(expr) << std::endl;
    return 0;
}
