#include "mainwindow.h"
#include "ui_mainwindow.h"

template<class F>
void foo(F f){
    qDebug() << f(10);
}

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    jointCanvas = new JointCanvas();

    QVBoxLayout *layout = new QVBoxLayout();
    layout->addWidget(jointCanvas);
    this->ui->widgetCanvasPlaceholder->setLayout(layout);

    /*imageDeformationWidget = new ImageDeformationWidget(imageDeformationController, imageViewport, &settings);
    QVBoxLayout *layout = new QVBoxLayout();
    QScrollArea *scrollArea = new QScrollArea(this);
    scrollArea->setWidget(imageViewport);
    layout->addWidget(imageDeformationWidget);
    layout->addWidget(scrollArea);
    ui->widget->setLayout(layout);*/



    auto lam = ([](int x){return x + 10;});
    foo(lam);

//    test01();
}

MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::on_pushButton_clicked()
{
    this->jointCanvas->test01();
//    this->test01();
}
