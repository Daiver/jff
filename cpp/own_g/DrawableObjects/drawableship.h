#ifndef DRAWABLESHIP_H
#define DRAWABLESHIP_H

#include <QImage>
#include "drawableobject.h"
#include "SpacePhysicalObjects/ship.h"

namespace DrawableObjects {

class DrawableShip : public DrawableObject
{
public:
    DrawableShip(SpacePhysicalObjects::Ship *ship);

    void draw(QPainter &painter);

private:
    SpacePhysicalObjects::Ship *ship;
    QImage shipImage;
};

}

#endif // DRAWABLESHIP_H
