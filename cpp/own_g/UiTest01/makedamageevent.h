#ifndef MAKEDAMAGEEVENT_H
#define MAKEDAMAGEEVENT_H

#include "gameevent.h"
#include "SpacePhysicalObjects/destroyablespaceobject.h"

class MakeDamageEvent : public GameEvent
{
public:
    MakeDamageEvent(
            const float damageAmount,
            SpacePhysicalObjects::DestroyableSpaceObject *obj2Damage):
        damageAmount(damageAmount),
        obj2Damage(obj2Damage)
    {

    }

    void processEvent(const float ticksPassed);

private:
    float damageAmount;
    SpacePhysicalObjects::DestroyableSpaceObject *obj2Damage;
};

#endif // MAKEDAMAGEEVENT_H
