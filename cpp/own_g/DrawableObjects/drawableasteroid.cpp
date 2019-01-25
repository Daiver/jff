#include "drawableasteroid.h"

#include "drawspacephysicalobjecttools.h"

DrawableObjects::DrawableAsteroid::DrawableAsteroid(SpacePhysicalObjects::Asteroid *asteroid):
    asteroid(asteroid),
    image(asteroid->size().toSize(), QImage::Format_RGBA8888)
{
    image.fill(Qt::transparent);
    QPainter innerPainter(&image);
    innerPainter.setBrush(Qt::red);
    innerPainter.drawEllipse(QRect(0, 0, asteroid->size().width() - 1, asteroid->size().height() - 1));
}

void DrawableObjects::DrawableAsteroid::draw(QPainter &painter)
{
    if(asteroid->isAlive())
        drawSpaceObjectByImage(*asteroid, image, painter);
}
