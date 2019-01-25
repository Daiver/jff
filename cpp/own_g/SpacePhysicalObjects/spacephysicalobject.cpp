#include "spacephysicalobject.h"

using namespace SpacePhysicalObjects;

void SpacePhysicalObject::setPosition(const QPointF &position)
{
    this->m_position = position;
}

void SpacePhysicalObject::setPosition(const float x, const float y)
{
    this->setPosition(QPointF(x, y));
}

void SpacePhysicalObject::setSize(const QSizeF &newSize)
{
    this->m_size = newSize;
}

void SpacePhysicalObject::setSize(const float width, const float height)
{
    this->setSize(QSizeF(width, height));
}

void SpacePhysicalObject::setOrientationAngle(const float newAngle)
{
    this->m_orientationAngle = newAngle;
}
