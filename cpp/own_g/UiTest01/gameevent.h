#ifndef GAMEEVENT_H
#define GAMEEVENT_H

class GameEvent
{
public:
    virtual ~GameEvent() {}
    virtual void processEvent(const float ticksPassed) = 0;
};

#endif // GAMEEVENT_H
