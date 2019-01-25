#ifndef DRAWABLEOBJECT_H
#define DRAWABLEOBJECT_H

#include <QPainter>

namespace DrawableObjects {

class DrawableObject
{
public:
    DrawableObject();

    virtual void draw(QPainter &painer) = 0;

    bool isVisible() const { return m_isVisible; }
    void setIsVisible(const bool isVisible)
    {
        this->m_isVisible = isVisible;
    }

protected:
    bool m_isVisible;
};

}

#endif // DRAWABLEOBJECT_H
