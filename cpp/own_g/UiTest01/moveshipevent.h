#ifndef MOVESHIPEVENT_H
#define MOVESHIPEVENT_H

#include "gameevent.h"
#include "gameloopworker.h"
#include "SpacePhysicalObjects/ship.h"

class MoveShipEvent : public GameEvent
{
public:
    MoveShipEvent(
            SpacePhysicalObjects::Ship *ship,
            const QPointF targetPosition):
        ship(ship),
        targetPosition(targetPosition)
    {

    }

    void processEvent(const float ticksPassed);

private:
    SpacePhysicalObjects::Ship *ship;
    QPointF targetPosition;    
};

#endif // MOVESHIPEVENT_H
