#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <iostream>
#include "casadi/casadi.hpp"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    const casadi::SX x = casadi::SX::sym("x");
    const casadi::SX y = casadi::SX::sym("y");
    const auto res = x*x + y;
    std::cout << res << std::endl;
}

MainWindow::~MainWindow()
{
    delete ui;
}
