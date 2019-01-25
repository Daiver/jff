#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QImage>
#include <QPainter>
#include <QDebug>
#include <QElapsedTimer>

#include "jointcanvas.h"
#include "particlefilter.h"
#include "joint.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

//    void drawJoints(QPainter *painter, QVector<Joint> &joints);
//    void test01();

    QVector<Joint> joints;

private slots:
    void on_pushButton_clicked();

private:
    Ui::MainWindow *ui;

    JointCanvas *jointCanvas;
};

#endif // MAINWINDOW_H
