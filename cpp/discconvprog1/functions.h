#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include "inference.h"

namespace DCP {

class Sum2 : public Function
{
public:
    static Monotonicity monotonicity(const int position, const ExpressionInfo &expr)
    {
        Q_UNUSED(expr);
        Q_UNUSED(position);
        return Monotonicity::Increasing;
    }

    static Sign sign(const std::vector<ExpressionInfo> &exprs)
    {
        Q_ASSERT(exprs.size() == 2);
        if(exprs[0].sign == exprs[1].sign)
            return exprs[0].sign;
        return Sign::Unknown;
    }

    static Curvature curvature(const std::vector<ExpressionInfo> &exprs)
    {
        Q_ASSERT(exprs.size() == 2);
        return Curvature::Affine;
    }
};

class Minus : public Function
{
public:
    static Monotonicity monotonicity(const int position, const ExpressionInfo &expr)
    {
        Q_UNUSED(expr);
        Q_UNUSED(position);
        return Monotonicity::Decreasing;
    }

    static Sign sign(const std::vector<ExpressionInfo> &exprs)
    {
        Q_ASSERT(exprs.size() == 1);
        if(exprs[0].sign == Sign::Negative)
            return Sign::Positive;
        if(exprs[0].sign == Sign::Positive)
            return Sign::Negative;
        return Sign::Unknown;
    }

    static Curvature curvature(const std::vector<ExpressionInfo> &exprs)
    {
        Q_ASSERT(exprs.size() == 1);
        return Curvature::Affine;
    }
};


class Minus2 : public Function
{
public:
    static Monotonicity monotonicity(const int position, const ExpressionInfo &expr)
    {
        Q_UNUSED(expr);
        if(position == 0)
            return Monotonicity::Increasing;
        return Monotonicity::Decreasing;
    }

    static Sign sign(const std::vector<ExpressionInfo> &exprs)
    {
        Q_ASSERT(exprs.size() == 2);
        if(exprs[0].sign == Sign::Positive && exprs[1].sign == Sign::Negative)
            return Sign::Positive;
        if(exprs[0].sign == Sign::Negative && exprs[1].sign == Sign::Positive)
            return Sign::Negative;
        return Sign::Unknown;
    }

    static Curvature curvature(const std::vector<ExpressionInfo> &exprs)
    {
        Q_ASSERT(exprs.size() == 2);
        return Curvature::Affine;
    }
};

//class Mul2 : public Function
//{
//public:
//    static Monotonicity monotonicity(const int position, const ExpressionInfo &expr)
//    {
//        Q_UNUSED(expr);
//        if(position == 0)
//            return Monotonicity::Increasing;
//        return Monotonicity::Decreasing;
//    }

//    static Sign sign(const std::vector<ExpressionInfo> &exprs)
//    {
//        Q_ASSERT(exprs.size() == 2);
//        if(exprs[0].sign == Sign::Positive && exprs[1].sign == Sign::Negative)
//            return Sign::Positive;
//        if(exprs[0].sign == Sign::Negative && exprs[1].sign == Sign::Positive)
//            return Sign::Negative;
//        return Sign::Unknown;
//    }

//    static Curvature curvature(const std::vector<ExpressionInfo> &exprs)
//    {
//        Q_ASSERT(exprs.size() == 2);
//        return Curvature::Affine;
//    }
//};


class Abs : public Function
{
public:
    static Monotonicity monotonicity(const int position, const ExpressionInfo &expr)
    {
        Q_UNUSED(expr);
        Q_ASSERT(position == 0);
        if(expr.sign == Sign::Negative)
            return Monotonicity::Decreasing;
        return Monotonicity::Increasing;
    }

    static Sign sign(const std::vector<ExpressionInfo> &exprs)
    {
        Q_ASSERT(exprs.size() == 1);
        Q_UNUSED(exprs)
        return Sign::Positive;
    }

    static Curvature curvature(const std::vector<ExpressionInfo> &exprs)
    {
        Q_UNUSED(exprs);
        return Curvature::Convex;
    }
};


}

#endif // FUNCTIONS_H
