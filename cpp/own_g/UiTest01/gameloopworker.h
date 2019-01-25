#ifndef GAMELOOPWORKER_H
#define GAMELOOPWORKER_H

#include <QObject>
#include <QTimer>
#include <QQueue>
#include <QVector>
#include "gameevent.h"
#include "entitycontroller.h"

class GameLoopWorker : public QObject
{
    Q_OBJECT
public:
    GameLoopWorker(QObject *parent);

    void start();

    void addEvent(GameEvent *event);
    void addEntityControlle(EntityController *entityController);

public slots:
    void run();
    void loopStep();

private:
    QTimer *gameLoopTimer;

    QVector<GameEvent *> events;
    QVector<GameEvent *> newEvents;
    QVector<EntityController *> entitieControlersToTrack;
};

#endif // GAMELOOPWORKER_H
