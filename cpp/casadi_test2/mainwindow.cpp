#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <iostream>
#include "casadi/casadi.hpp"
#include "casadi/core/calculus.hpp"


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    using namespace casadi;

    const SX vertices = SX::sym("vertices", 3, 2);    
    SX x2 = vertices;
    x2 = x2 + x2;

    std::cout << vertices << std::endl;
    std::cout << x2 << std::endl;
}

MainWindow::~MainWindow()
{
    delete ui;
}
