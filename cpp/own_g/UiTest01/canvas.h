#ifndef CANVAS_H
#define CANVAS_H

#include <QWidget>
#include <QPaintEvent>
#include <QVector>

#include "DrawableObjects/drawableobject.h"

class Canvas: public QWidget
{
    Q_OBJECT
public:
    explicit Canvas(QWidget *parent = 0);

    void addDrawableObject(DrawableObjects::DrawableObject *obj);

signals:
    void mousePressed(QMouseEvent *event);

protected:
    void paintEvent(QPaintEvent *event);
    void mousePressEvent(QMouseEvent *event);

private:
    bool m_printElapsed;

    QVector<DrawableObjects::DrawableObject *> objects;

};

#endif // CANVAS_H
