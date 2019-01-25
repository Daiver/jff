#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QThread>
#include <QLayout>
#include <QTimer>
#include <QMouseEvent>
#include <QDebug>

#include "SpacePhysicalObjects/ship.h"
#include "DrawableObjects/drawableship.h"
#include "DrawableObjects/drawableplanet.h"
#include "moveshipevent.h"
#include "SpacePhysicalObjects/asteroid.h"
#include "DrawableObjects/drawableasteroid.h"
#include "makedamageevent.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    canvas = new Canvas();
    this->ui->centralWidget->layout()->addWidget(canvas);
    QObject::connect(canvas, SIGNAL(mousePressed(QMouseEvent*)), this, SLOT(on_canvasClick(QMouseEvent*)));

    planet = new SpacePhysicalObjects::Planet();
    planet->setSize(QSizeF(60, 60));
    planet->setPosition(QPointF(100, 100));
    DrawableObjects::DrawablePlanet *drPlanet = new DrawableObjects::DrawablePlanet(planet);
    canvas->addDrawableObject(drPlanet);    

    asteroid = new SpacePhysicalObjects::Asteroid();
    asteroid->setPosition(QPointF(300, 110));
    asteroid->setSize(QSizeF(30, 30));
    asteroid->setMaxHealthAndHealth(100);
    DrawableObjects::DrawableAsteroid *drAsteroid = new DrawableObjects::DrawableAsteroid(asteroid);
    canvas->addDrawableObject(drAsteroid);

    ship = new SpacePhysicalObjects::Ship();
    ship->setSize(QSizeF(59, 65) / 2);
    ship->setPosition(QPointF(50, 50));
    ship->setOrientationAngle(90);
    ship->setSpeed(15.0);
    ship->setMaxHealthAndHealth(100);
    DrawableObjects::DrawableShip *drShip = new DrawableObjects::DrawableShip(ship);
    canvas->addDrawableObject(drShip);    

    QTimer *renderTimer = new QTimer(this);
    connect(renderTimer, SIGNAL(timeout()), canvas, SLOT(update()));
    renderTimer->start(1.0/60.0);

    gameLoopWorker = new GameLoopWorker(0);
    gamerShipController = new GamerShipController(gameLoopWorker, ship);
    gameLoopWorker->addEntityControlle(gamerShipController);

    QThread *somethread = new QThread();
    gameLoopWorker->moveToThread(somethread);
    QTimer::singleShot(0, gameLoopWorker, SLOT(run()));
    somethread->start();
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_pushButton_clicked()
{
    ship->setPosition(ship->position() + QPointF(10, 10));
}

void MainWindow::on_canvasClick(QMouseEvent *event)
{
    if(event->button() == Qt::LeftButton){
        gameLoopWorker->addEvent(new MakeDamageEvent(10, asteroid));
    }
    if(event->button() != Qt::RightButton)
        return;
    qDebug() << event->pos();
    gamerShipController->setMoveTarget(event->pos());
}
