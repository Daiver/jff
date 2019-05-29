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
    const SX targets = SX::sym("targets", 3, 2);

    const auto res = vertices - targets;
    const SX jacobian = SX::jacobian(res, vertices);

    const Sparsity sparsity = jacobian.sparsity();
//    sparsity.get_triplet()

    std::cout << res << std::endl;
    std::cout << bool(jacobian.is_dense()) << std::endl;
    std::cout << jacobian << std::endl;
}

MainWindow::~MainWindow()
{
    delete ui;
}
