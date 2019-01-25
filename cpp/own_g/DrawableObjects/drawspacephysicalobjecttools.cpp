#include "drawspacephysicalobjecttools.h"

QPointF DrawableObjects::upLeftCornetOfSpaceObject(const SpacePhysicalObjects::SpacePhysicalObject &object)
{
    const QSizeF halfSize = object.size() / 2;
    const QPointF position = object.position();
    return position - QPointF(halfSize.width(), halfSize.height());
}

QTransform DrawableObjects::transformForSpaceObject(const QPointF &position, const QSizeF &size, const float angle)
{
    QTransform trans;
    trans.translate(position.x(), position.y());
    trans.rotate(angle);
    trans.translate(-size.width() / 2, -size.height() / 2);
    return trans;
}

void DrawableObjects::drawSpaceObjectByImage(const SpacePhysicalObjects::SpacePhysicalObject &obj, const QImage &image, QPainter &painter)
{
    QTransform trans = transformForSpaceObject(obj.position(), obj.size(), obj.orientationAngle());
    trans.scale(obj.size().width() / (float)image.width(), obj.size().height() / (float)image.height());
    painter.setTransform(trans);
    painter.drawImage(QPoint(0, 0), image);
    painter.resetTransform();
}
