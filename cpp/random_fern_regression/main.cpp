#include <iostream>

#include "Eigen/Core"
#include "fern.h"
#include "ferntrain.h"
#include "boosting.h"
#include "eigexport.h"

using namespace RandomFern;

typedef Eigen::Vector2f Output;

/*Output toEig(const float f)*/
//{
    //Eigen::Matrix<float, 1, 1> res;
    //res[0] = f;
    //return res;
//}

//Eigen::Matrix<Output, -1, 1> toEig(const Eigen::MatrixXf &vec)
//{
    //assert(vec.cols() == 1);
    //Eigen::Matrix<Output, -1, 1> res(vec.rows());

    //for(int i = 0; i < res.rows(); ++i)
        //res(i, 0) = toEig(vec(i, 0));

    //return res;
//}

Eigen::Matrix<std::pair<float, float>, -1, 1> computeBorders(const Eigen::MatrixXf &data)
{
    Eigen::Matrix<std::pair<float, float>, -1, 1> res(data.cols());
    const Eigen::VectorXf maxs = data.array().colwise().maxCoeff();
    const Eigen::VectorXf mins = data.array().colwise().minCoeff();
    for(int i = 0; i < data.cols(); ++i)
        res(i, 0) = std::make_pair(mins[i], maxs[i]);
    
    return res;
}

inline Output l2Grad(const Output &residual)
{
    return residual;
}

//inline Output l1Grad(const Output &residual)
//{
    //return toEig(fabs(residual[0])/residual[0]);
//}

//inline Output huberGrad(const Output &residual)
//{
    //const float delta = 2.0;
    //const float absVal = fabs(residual[0]);
    //if(absVal < delta)
        //return residual;
    //return toEig(delta * absVal / residual[0]);
//}

int nearestPointInd(const Eigen::Vector2f &point, const Eigen::Matrix<float, -1, 2> &points)
{
    int res = 0;
    float minDist = 1e8;
    for(int i = 0; i < points.rows(); ++i){
        const float dist = (point - points.row(i).transpose()).norm();
        if(dist < minDist){
            minDist = dist;
            res = i;
        }
    }
    return res;
}

Eigen::VectorXf featureVectorForShape(
        const Eigen::Matrix<float, -1, 2> &targetShape, 
        const Eigen::Matrix<float, -1, 2> &floatingShape,
        const Eigen::Vector2f &translate)
{
    Eigen::VectorXf res(floatingShape.rows() * 2);
    for(int i = 0; i < floatingShape.rows(); ++i){
        const Eigen::Vector2f point = floatingShape.row(i).transpose() + translate;
        const int nearest = nearestPointInd(point, targetShape);
        const Eigen::Vector2f diff = targetShape.row(nearest).transpose() - point;
        res(2 * i + 0, 0) = diff[0];
        res(2 * i + 1, 0) = diff[1];
    }
    return res;
}

int main()
{
    srand(42);
    
    return 0;
}

