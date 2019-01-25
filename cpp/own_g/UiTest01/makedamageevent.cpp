#include "makedamageevent.h"

#include <QDebug>

void MakeDamageEvent::processEvent(const float ticksPassed)
{
    Q_UNUSED(ticksPassed);
    this->obj2Damage->getDamage(this->damageAmount);
    qDebug() << "Damaged! Health" << obj2Damage->currentHealth();
}
