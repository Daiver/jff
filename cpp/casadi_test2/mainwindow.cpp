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

    const casadi::SX x = SX::sym("x", 2, 3);
//    const casadi::SX y = SX::sym("y");
    const auto xc1 = x(Slice(), 1);
    const auto xc2 = x(Slice(), 2);
    const auto res = SX::dot(xc1, xc2);

    const auto resFunc = Function("resFunc", {x}, {res});

//    const auto jac = SX::simplify(SX::jacobian(res, x));
//    const auto jac = SX::simplify(SX::jacobian(res, y));
    std::cout << x << std::endl;
    std::cout << xc1 << std::endl;
    std::cout << xc2 << std::endl;
    std::cout << res << std::endl;    
    std::cout << resFunc << std::endl;
    const auto xVal = DM({{1, 2, 3}});
    resFunc(xVal);
//    std::cout << jac << std::endl;
}

MainWindow::~MainWindow()
{
    delete ui;
}
