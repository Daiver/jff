#-------------------------------------------------
#
# Project created by QtCreator 2015-10-02T23:39:30
#
#-------------------------------------------------

QT       += core gui

QMAKE_CXXFLAGS += -std=c++14

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = IK-fun
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    joint.cpp \
    particlefilter.cpp \
    jointcanvas.cpp

HEADERS  += mainwindow.h \
    joint.h \
    particlefilter.h \
    jointcanvas.h

FORMS    += mainwindow.ui
