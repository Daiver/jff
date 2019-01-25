#include "gameloopworker.h"

#include <QDebug>

GameLoopWorker::GameLoopWorker(QObject *parent):
    QObject(parent)
{

}

void GameLoopWorker::start()
{
    gameLoopTimer = new QTimer(this);
    gameLoopTimer->setInterval(100);
    connect(gameLoopTimer, SIGNAL(timeout()), this, SLOT(loopStep()));
    gameLoopTimer->start();
}

void GameLoopWorker::addEvent(GameEvent *event)
{
    this->newEvents.append(event);
}

void GameLoopWorker::addEntityControlle(EntityController *entityController)
{
    this->entitieControlersToTrack.append(entityController);
}

void GameLoopWorker::run()
{
    this->start();
}

void GameLoopWorker::loopStep()
{
    for(int i = 0; i < entitieControlersToTrack.size(); ++i){
        const float ticksPassed = 1;
        entitieControlersToTrack[i]->processTick(ticksPassed);
    }

    for(int i = 0; i < events.size(); ++i){
        const float ticksPassed = 1;
        events[i]->processEvent(ticksPassed);
        delete events[i];
    }    
    events = newEvents;
    newEvents.clear();
}
