#ifndef INFERENCE_H
#define INFERENCE_H

#include <vector>
#include "enums.h"
#include "expressioninfo.h"
#include "function.h"

namespace DCP {

template<typename Function>
inline ExpressionInfo inference(const std::vector<ExpressionInfo> &args)
{
    const Sign resSign = Function::sign(args);
    const Curvature curv = Function::curvature(args);

    if(curv == Curvature::Constant)
        return ExpressionInfo{resSign, Curvature::Constant};
    bool allArgsAreConstant = true;;
    for(unsigned int i = 0; i < args.size(); ++i){
        if(args[i].curvature != Curvature::Constant){
            allArgsAreConstant = false;
            break;
        }
    }
    if(allArgsAreConstant)
        return ExpressionInfo{resSign, Curvature::Constant};
    bool isResConvex = false;

    if(isConvexOrAffine(curv)){
        bool isConvexConditionsHolds = true;
        for(unsigned int i = 0; i < args.size(); ++i){
            const Monotonicity monotonicity = Function::monotonicity(i, args[i]);
            Q_ASSERT(monotonicity != Monotonicity::Unknown);//I don't know what to do =)
            const Curvature argCurv = args[i].curvature;
            const bool isConvexAndIncr  = monotonicity == Monotonicity::Increasing && isConvexOrAffine(argCurv);
            const bool isConcaveAndDecr = monotonicity == Monotonicity::Decreasing && isConcaveOrAffine(argCurv);
            const bool isArgAffine      = argCurv == Curvature::Affine;
            isConvexConditionsHolds = isConvexConditionsHolds && (isConcaveAndDecr || isConvexAndIncr || isArgAffine);
        }
        isResConvex = isConvexConditionsHolds;
    }

    bool isResConcave = false;
    if(isConcaveOrAffine(curv)){
        bool isConcaveConditionsHolds = true;
        for(unsigned int i = 0; i < args.size(); ++i){
            const Monotonicity monotonicity = Function::monotonicity(i, args[i]);
            Q_ASSERT(monotonicity != Monotonicity::Unknown);//I don't know what to do =)
            const Curvature argCurv = args[i].curvature;
            const bool isConcaveAndIncr = monotonicity == Monotonicity::Increasing && isConcaveOrAffine(argCurv);
            const bool isConvexAndDecr  = monotonicity == Monotonicity::Decreasing && isConvexOrAffine(argCurv);
            const bool isArgAffine      = argCurv == Curvature::Affine;
            isConcaveConditionsHolds    = isConcaveConditionsHolds && (isConvexAndDecr || isConcaveAndIncr || isArgAffine);
        }
        isResConcave = isConcaveConditionsHolds;
    }

    Curvature resCurv = Curvature::Unknown;
    if(isResConcave)
        resCurv = Curvature::Concave;
    if(isResConvex)
        resCurv = Curvature::Convex;
    if(isResConcave && isResConvex)
        resCurv = Curvature::Affine;
    return ExpressionInfo{resSign, resCurv};
}

}

#endif // INFERENCE_H
