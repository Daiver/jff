#ifndef DRAWSPACEPHYSICALOBJECTTOOLS_H
#define DRAWSPACEPHYSICALOBJECTTOOLS_H

#include <QPainter>
#include <QTransform>
#include <QImage>
#include "SpacePhysicalObjects/spacephysicalobject.h"

namespace DrawableObjects {

QPointF upLeftCornetOfSpaceObject(const SpacePhysicalObjects::SpacePhysicalObject &object);
QTransform transformForSpaceObject(const QPointF &position, const QSizeF &size, const float angle);
void drawSpaceObjectByImage(const SpacePhysicalObjects::SpacePhysicalObject &obj, const QImage &image, QPainter &painter);

}

#endif // DRAWSPACEPHYSICALOBJECTTOOLS_H
