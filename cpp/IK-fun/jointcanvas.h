#ifndef JOINTCANVAS_H
#define JOINTCANVAS_H

#include <QWidget>
#include <QLabel>
#include <QPainter>
#include <QImage>
#include <QElapsedTimer>
#include <QDebug>
#include <QMouseEvent>

#include "joint.h"
#include "particlefilter.h"

class JointCanvas : public QLabel
{
    Q_OBJECT
public:
    JointCanvas(QWidget *parent = NULL);

    QVector<Joint> joints;

    void draw();
    void test01();
    void drawJoints(QPainter *painter, QVector<Joint> &joints);

    Eigen::Vector2f target;

protected:
    void mousePressEvent (QMouseEvent * ev ) ;
};

#endif // JOINTCANVAS_H

