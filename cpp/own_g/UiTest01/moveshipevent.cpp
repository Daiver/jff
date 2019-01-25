#include "moveshipevent.h"

#include <QVector2D>
//#include <QDebug>

void MoveShipEvent::processEvent(const float ticksPassed)
{
    Q_UNUSED(ticksPassed);
    const QVector2D zeroRotationDirection(0, 1);
    const QVector2D directionUnnorm = QVector2D(targetPosition - ship->position());
    const float dist2Target         = directionUnnorm.length();
    const QVector2D directionNorm   = directionUnnorm / dist2Target;

    const float angle = atan2(zeroRotationDirection.y(), zeroRotationDirection.x()) + atan2(directionNorm.y(), directionNorm.x());
    if(fabs(dist2Target) > 0.00001)
        ship->setOrientationAngle(angle * 180 / M_PI);
    if(dist2Target < ship->speed()){
        ship->setPosition(targetPosition);
        return;
    }

    ship->setPosition(ship->position() + (ship->speed() * directionNorm).toPointF());
}
