#ifndef ENTITYCONTROLLER_H
#define ENTITYCONTROLLER_H

class EntityController
{
public:
    virtual void processTick(const float ticksPassed) = 0;
};

#endif // ENTITYCONTROLLER_H
