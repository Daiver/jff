#ifndef JOINT_H
#define JOINT_H

#include <QMatrix3x3>
#include <QVector2D>
#include <QVector3D>
#include <qmath.h>

#include <eigen3/Eigen/Dense>

Eigen::Vector3f v3ffromv2f(const Eigen::Vector2f &vec);
Eigen::Vector2f v2ffromv3f(const Eigen::Vector3f &vec);
QVector2D qv2dfromv2f(const Eigen::Vector2f &vec);
Eigen::Vector2f v2ffromqv2d(const QVector2D &vec);


class Joint
{
public:
    Joint();
    Joint(QVector<Joint> *joints, int parentInd, float angle, const Eigen::Vector2f &translate);

    Eigen::Vector2f getOwnGlobalPosition() const;
    Eigen::Matrix3f getOwnTransformation() const;

    QVector<Joint> *joints;

    int parentInd;

    float angle;
    Eigen::Vector2f translate;
};

void copyJointVector(const QVector<Joint> &src, QVector<Joint> *dst);
void applyAnglesToSystem(QVector<Joint> &joints, const Eigen::VectorXf &params);






inline Eigen::Vector2f Joint::getOwnGlobalPosition() const
{
    if(parentInd == -1)
        return this->translate;
    const Joint *parent = &(this->joints->at(parentInd));
    return v2ffromv3f(parent->getOwnTransformation() * v3ffromv2f(this->translate));
}

inline Eigen::Matrix3f Joint::getOwnTransformation() const
{
    Eigen::Matrix3f rotM = Eigen::Matrix3f::Zero();
    float cs = cos(this->angle);
    float sn = sin(this->angle);
    rotM(0, 0) =  cs;
    rotM(0, 1) = -sn;
    rotM(1, 0) =  sn;
    rotM(1, 1) =  cs;
    rotM(2, 2) =  1;

    Eigen::Matrix3f transM = Eigen::Matrix3f::Zero();
    transM(0, 0) = 1;
    transM(1, 1) = 1;
    transM(2, 2) = 1;
    transM(0, 2) = this->translate[0];
    transM(1, 2) = this->translate[1];

    if(parentInd == -1)
        return transM * rotM;

    const Joint *parent = &(this->joints->at(parentInd));

    return parent->getOwnTransformation() * transM * rotM;
}

inline void applyAnglesToSystem(QVector<Joint> &joints, const Eigen::VectorXf &params)
{
    Q_ASSERT(joints.size() == params.rows());
    for(int i = 0; i < joints.size(); ++i)
        joints[i].angle = params[i];
}


#endif // JOINT_H
