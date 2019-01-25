#ifndef DESTROYABLESPACEOBJECT_H
#define DESTROYABLESPACEOBJECT_H

#include "spacephysicalobject.h"

namespace SpacePhysicalObjects {

class DestroyableSpaceObject : public SpacePhysicalObject
{
public:
    DestroyableSpaceObject();

    virtual float currentHealth() const { return m_health; }
    virtual void setCurrentHealth(const float health);

    virtual float maxHealth() const { return m_maxHealth; }
    virtual void setMaxHealth(const float maxHealth) { this->m_maxHealth = maxHealth; }

    virtual bool isAlive() const { return m_isAlive; }

    virtual void getDamage(const float dmg);

    void setMaxHealthAndHealth(const float maxHealth);

protected:
    float m_health;
    float m_maxHealth;
    bool m_isAlive;

    void checkIsAlive();

    virtual void onDead();
};

}

#endif // DESTROYABLESPACEOBJECT_H
