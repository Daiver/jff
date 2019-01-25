#ifndef SHIP_H
#define SHIP_H

#include "spacephysicalobject.h"
#include "destroyablespaceobject.h"

namespace SpacePhysicalObjects {

class Ship : public DestroyableSpaceObject
{
public:    
    Ship();
    float speed() const { return m_speed; }
    void setSpeed(const float speed)
    {
        this->m_speed = speed;
    }

private:
    float m_speed;
};

}

#endif // SHIP_H
