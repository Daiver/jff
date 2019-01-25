QT += core testlib
QT -= gui

#CONFIG += c++11
QMAKE_CXXFLAGS += -std=c++17

TARGET = gbt
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app


INCLUDEPATH += /home/daiver/coding/eigen/
INCLUDEPATH += /home/daiver/coding/range-v3/include/

SOURCES += main.cpp \
    stump.cpp \
    stumptests.cpp \
    targetlossinfomse.cpp \
    nodeleaf.cpp \
    nodedecision.cpp \
    nodebase.cpp \
    splittinginfo.cpp

HEADERS += \
    stump.h \
    stumptests.h \
    targetlossinfomse.h \
    nodeleaf.h \
    nodedecision.h \
    nodebase.h \
    splittinginfo.h
