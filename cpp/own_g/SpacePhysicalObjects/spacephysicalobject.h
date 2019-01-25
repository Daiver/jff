#ifndef SPACEPHYSICALOBJECT_H
#define SPACEPHYSICALOBJECT_H

#include <QSizeF>
#include <QPointF>

namespace SpacePhysicalObjects {

class SpacePhysicalObject
{
public:
    virtual ~SpacePhysicalObject() {}

    virtual QPointF position() const { return m_position; }
    virtual void setPosition(const QPointF &position);
    virtual void setPosition(const float x, const float y);

    virtual QSizeF size() const { return m_size; }
    virtual void setSize(const QSizeF &newSize);
    virtual void setSize(const float width, const float height);

    virtual float orientationAngle() const { return m_orientationAngle; }
    virtual void setOrientationAngle(const float newAngle);

private:
    QPointF m_position;
    QSizeF m_size;
    float m_orientationAngle;
};

}

#endif // SPACEPHYSICALOBJECT_H
