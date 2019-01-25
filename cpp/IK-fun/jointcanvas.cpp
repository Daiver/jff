#include "jointcanvas.h"

JointCanvas::JointCanvas(QWidget *parent): QLabel(parent)
{
    QVector<Joint> joints;
    joints.resize(5);
    joints[0] = Joint(&joints, -1,  M_PI/4.0, v2ffromqv2d(QVector2D(100, 200)));
    joints[1] = Joint(&joints,  0, -M_PI/4.0, v2ffromqv2d(QVector2D(50, 00)));
    joints[2] = Joint(&joints,  1,  M_PI/4.0, v2ffromqv2d(QVector2D(50, 00)));
    joints[3] = Joint(&joints,  2, -M_PI/3.0, v2ffromqv2d(QVector2D(50, 00)));
    joints[4] = Joint(&joints,  3,  0       , v2ffromqv2d(QVector2D(50, 00)));

    copyJointVector(joints, &this->joints);

    target << 256, 334;
}

void JointCanvas::draw()
{

}

void JointCanvas::drawJoints(QPainter *painter, QVector<Joint> &joints)
{
    for(int i = 0; i < joints.size(); ++i){
        Joint &joint = joints[i];
        QPoint jointPoint = qv2dfromv2f(joint.getOwnGlobalPosition()).toPoint();
        if(joint.parentInd != -1){
            QPoint jointPoint2 =  qv2dfromv2f(joints[joint.parentInd].getOwnGlobalPosition()).toPoint();
            painter->drawLine(jointPoint, jointPoint2);
        }

        painter->drawEllipse(jointPoint, 10, 10);
    }
}

void JointCanvas::mousePressEvent(QMouseEvent *ev)
{
    target << ev->x() , ev->y();
    this->test01();
}

void JointCanvas::test01()
{
    srand(time(NULL));

    QImage canvas(this->width(), this->height(), QImage::Format_RGB888);
    canvas.fill(Qt::white);
    QPainter painter(&canvas);
//    painter.drawEllipse(50, 50, 10, 10);

    painter.setBrush(QBrush(Qt::green));
    this->drawJoints(&painter, joints);

    qDebug() << qv2dfromv2f(joints.last().getOwnGlobalPosition());


//    target << 250, 250;

    auto costF = [&](const Eigen::VectorXf &params){
        //QVector<Joint> joints;
        //copyJointVector(this->joints, &joints);
        applyAnglesToSystem(joints, params);
        Eigen::VectorXf res = joints.last().getOwnGlobalPosition();
        return (res - target).array().abs().sum();
    };

    Eigen::VectorXf curr = Eigen::VectorXf(5);
    curr << M_PI/4.0, -M_PI/4.0, M_PI/4.0, -M_PI/3.0, 0;
    qDebug() << costF(curr);

    QElapsedTimer timer;
    timer.start();
    Eigen::VectorXf newParams = ParticleFilter::processParticleFilter(
                costF,
//                QVector<QPair<float, float>>(joints.size(), qMakePair(-M_PI/0.7, M_PI/0.7)),
                QVector<QPair<float, float>>(joints.size(), qMakePair(-M_PI/4, M_PI/4)),
                50000, 20, 1.0);

    qDebug() << "Elapsed" << timer.elapsed()/1000.0;
    applyAnglesToSystem(joints, newParams);

    painter.setBrush(QBrush(Qt::red));
    this->drawJoints(&painter, joints);

    painter.setBrush(QBrush(Qt::blue));
    painter.drawEllipse(qv2dfromv2f(target).toPoint(), 5, 5);

    this->setPixmap(QPixmap::fromImage(canvas));

}
