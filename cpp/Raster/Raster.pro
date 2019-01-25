QT += core
#QT -= gui

CONFIG += c++11

TARGET = Raster
CONFIG += console
CONFIG -= app_bundle

INCLUDEPATH += /usr/include/eigen3/

TEMPLATE = app

SOURCES += main.cpp \
    objimportexport.cpp

HEADERS += \
    objimportexport.h
