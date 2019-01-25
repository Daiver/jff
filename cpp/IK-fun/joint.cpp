#include "joint.h"

Joint::Joint(): parentInd(NULL), translate(Eigen::Vector2f::Zero()), angle(0), joints(NULL)
{

}

Joint::Joint(QVector<Joint> *joints, int parent, float angle, const Eigen::Vector2f &translate): angle(angle), translate(translate), parentInd(parent), joints(joints)
{

}

Eigen::Vector3f v3ffromv2f(const Eigen::Vector2f &vec)
{
    Eigen::Vector3f res;
    res << vec[0], vec[1], 1.0;
    return res;
}


Eigen::Vector2f v2ffromv3f(const Eigen::Vector3f &vec)
{
    Eigen::Vector2f res;
    res << vec[0], vec[1];
    return res;
}


QVector2D qv2dfromv2f(const Eigen::Vector2f &vec)
{
    return QVector2D(vec[0], vec[1]);
}


Eigen::Vector2f v2ffromqv2d(const QVector2D &vec)
{
    Eigen::Vector2f res;
    res << vec[0], vec[1];
    return res;
}



void copyJointVector(const QVector<Joint> &src, QVector<Joint> *dst)
{
    dst->resize(src.size());
    for(int i = 0; i < src.size(); ++i){
        (*dst)[i] = src[i];
        (*dst)[i].joints = dst;
    }
}



Eigen::VectorXf anglesToParams(const QVector<Joint> &joints)
{
    Eigen::VectorXf res(joints.size());
    for(int i = 0; i < joints.size(); ++i)
        res[i] = joints[i].angle;
    return res;
}
