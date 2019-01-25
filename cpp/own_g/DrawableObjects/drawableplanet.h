#ifndef DRAWABLEPLANET_H
#define DRAWABLEPLANET_H

#include <QImage>
#include "drawableobject.h"
#include "SpacePhysicalObjects/planet.h"

namespace DrawableObjects {

class DrawablePlanet : public DrawableObject
{
public:
    DrawablePlanet(SpacePhysicalObjects::Planet *planet);

    void draw(QPainter &painer);

private:
    SpacePhysicalObjects::Planet *planet;

    QImage planetImage;
};

}

#endif // DRAWABLEPLANET_H
