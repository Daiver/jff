#ifndef ASTEROID_H
#define ASTEROID_H

#include "spacephysicalobject.h"
#include "destroyablespaceobject.h"

namespace SpacePhysicalObjects {

class Asteroid : public DestroyableSpaceObject
{
public:
    Asteroid();

protected:
    void onDead();
};

}

#endif // ASTEROID_H
