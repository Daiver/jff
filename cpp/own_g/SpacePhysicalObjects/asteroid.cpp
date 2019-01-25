#include "asteroid.h"

#include <QDebug>

SpacePhysicalObjects::Asteroid::Asteroid(): DestroyableSpaceObject()
{

}

void SpacePhysicalObjects::Asteroid::onDead()
{
    qDebug() << "asteroid is dead";
}
