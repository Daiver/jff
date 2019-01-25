#include "drawableplanet.h"

#include "drawspacephysicalobjecttools.h"

using namespace DrawableObjects;

DrawablePlanet::DrawablePlanet(SpacePhysicalObjects::Planet *planet):
    planet(planet),
    planetImage(planet->size().toSize(), QImage::Format_RGBA8888)
{    
    planetImage.fill(Qt::transparent);
    QPainter innerPainter(&planetImage);
    innerPainter.setBrush(QColor("#87ceeb"));
    innerPainter.drawEllipse(QRect(0, 0, planet->size().width() - 1, planet->size().height() - 1));
}

void DrawablePlanet::draw(QPainter &painter)
{
    drawSpaceObjectByImage(*planet, planetImage, painter);
}
