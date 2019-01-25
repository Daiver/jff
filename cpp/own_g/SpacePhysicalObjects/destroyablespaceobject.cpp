#include "destroyablespaceobject.h"

SpacePhysicalObjects::DestroyableSpaceObject::DestroyableSpaceObject():
    m_isAlive(true)
{

}

void SpacePhysicalObjects::DestroyableSpaceObject::setCurrentHealth(const float health)
{
    this->m_health = health;
    this->checkIsAlive();
}

void SpacePhysicalObjects::DestroyableSpaceObject::getDamage(const float dmg)
{
    const float curHealth = currentHealth();
    if(curHealth > 0)
        this->setCurrentHealth(curHealth - dmg);
}

void SpacePhysicalObjects::DestroyableSpaceObject::setMaxHealthAndHealth(const float maxHealth)
{
    this->setMaxHealth(maxHealth);
    this->setCurrentHealth(maxHealth);
}

void SpacePhysicalObjects::DestroyableSpaceObject::checkIsAlive()
{
    if(!isAlive())
        return;
    if(this->currentHealth() <= 0){
        this->m_isAlive = false;
        this->onDead();
    }
}

void SpacePhysicalObjects::DestroyableSpaceObject::onDead()
{

}
