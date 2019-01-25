#include "gamershipcontroller.h"

#include <QVector2D>
#include "moveshipevent.h"

GamerShipController::GamerShipController(GameLoopWorker *loopWorker, SpacePhysicalObjects::Ship *ship): loopWorker(loopWorker), ship(ship)
{
    this->setMoveTarget(ship->position());
}

void GamerShipController::processTick(const float ticksPassed)
{
    Q_UNUSED(ticksPassed);
    const QVector2D diff = QVector2D(ship->position() - targetPoint);
    if(diff.length() > 0.000001)
        loopWorker->addEvent(new MoveShipEvent(ship, targetPoint));
}

void GamerShipController::setMoveTarget(const QPointF &targetPoint)
{
    this->targetPoint = targetPoint;
}
