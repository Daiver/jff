#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QMouseEvent>
#include "canvas.h"
#include "SpacePhysicalObjects/ship.h"
#include "SpacePhysicalObjects/planet.h"
#include "SpacePhysicalObjects/asteroid.h"
#include "gamershipcontroller.h"
#include "gameloopworker.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
    void on_pushButton_clicked();

    void on_canvasClick(QMouseEvent *event);

private:
    Ui::MainWindow *ui;

    Canvas *canvas;

    SpacePhysicalObjects::Ship *ship;
    SpacePhysicalObjects::Planet *planet;
    SpacePhysicalObjects::Asteroid *asteroid;
    GamerShipController *gamerShipController;
    GameLoopWorker *gameLoopWorker;
};

#endif // MAINWINDOW_H
