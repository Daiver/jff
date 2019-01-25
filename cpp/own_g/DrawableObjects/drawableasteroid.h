#ifndef DRAWABLEASTEROID_H
#define DRAWABLEASTEROID_H

#include "SpacePhysicalObjects/asteroid.h"
#include "drawableobject.h"

namespace DrawableObjects {

class DrawableAsteroid : public DrawableObject
{
public:
    DrawableAsteroid(SpacePhysicalObjects::Asteroid *asteroid);
    void draw(QPainter &painter);
private:
    SpacePhysicalObjects::Asteroid *asteroid;
    QImage image;
};

}
#endif // DRAWABLEASTEROID_H
