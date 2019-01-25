#ifndef GAMERSHIPCONTROLLER_H
#define GAMERSHIPCONTROLLER_H

#include "SpacePhysicalObjects/ship.h"
#include "entitycontroller.h"
#include "gameloopworker.h"

class GamerShipController : public EntityController
{
public:
    GamerShipController(GameLoopWorker *loopWorker, SpacePhysicalObjects::Ship *ship);

    void processTick(const float ticksPassed);

    void setMoveTarget(const QPointF &targetPoint);

private:
    GameLoopWorker *loopWorker;
    SpacePhysicalObjects::Ship *ship;

    QPointF targetPoint;
};

#endif // GAMERSHIPCONTROLLER_H
