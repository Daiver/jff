#include "drawableship.h"

#include "drawspacephysicalobjecttools.h"

using namespace DrawableObjects;

DrawableShip::DrawableShip(SpacePhysicalObjects::Ship *ship):
    DrawableObject(),
    ship(ship),
    shipImage(QImage(ship->size().toSize(), QImage::Format_RGBA8888))
{    
    const QString path2Sprite = "../sprites/ship1.png";
    shipImage = QImage(path2Sprite);
}

void DrawableShip::draw(QPainter &painter)
{    
    if(ship->isAlive())
        drawSpaceObjectByImage(*ship, shipImage, painter);
}
