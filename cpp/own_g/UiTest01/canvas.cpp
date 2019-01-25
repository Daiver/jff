#include "canvas.h"
#include <QPainter>
#include <QElapsedTimer>
#include <QDebug>

Canvas::Canvas(QWidget *parent) : QWidget(parent),
    m_printElapsed(false)
{

}

void Canvas::addDrawableObject(DrawableObjects::DrawableObject *obj)
{
    this->objects.append(obj);
}

void Canvas::paintEvent(QPaintEvent *event)
{
    Q_UNUSED(event);
    QPainter painter(this);

    QElapsedTimer timer;
    timer.start();

    foreach (DrawableObjects::DrawableObject *obj, objects) {
        if(obj->isVisible())
            obj->draw(painter);
    }

//    painter.drawEllipse(QPoint(200, 200), 10, 10);

    if(m_printElapsed)
        qDebug() << "Render" << timer.elapsed() / 1000.0;
}

void Canvas::mousePressEvent(QMouseEvent *event)
{
    emit mousePressed(event);
}
