QT -= gui

CONFIG += c++14 console
CONFIG -= app_bundle
DEFINES += QT_DEPRECATED_WARNINGS

QMAKE_CXXFLAGS = -g -O0 -pipe -Wparentheses -Wreturn-type -Wcast-qual -Wall -Wpointer-arith -Wwrite-strings -Wconversion -Wno-unknown-pragmas -Wno-long-long   -DIPOPT_BUILD
INCLUDEPATH += /home/daiver/Downloads/Ipopt-3.12.11/build/include/coin
LIBS += \
    -L/home/daiver/Downloads/Ipopt-3.12.11/build/lib -lipopt \
    -L/usr/lib/gcc/x86_64-linux-gnu/7 \
    -L/usr/lib/gcc/x86_64-linux-gnu/7/../../../x86_64-linux-gnu \
    -L/usr/lib/gcc/x86_64-linux-gnu/7/../../../../lib -L/lib/../lib \
    -L/usr/lib/../lib -L/usr/lib/gcc/x86_64-linux-gnu/7/../../.. \
    -llapack -lblas -lm -ldl -lcoinmumps -lblas -lgfortran -lm -lquadmath


#QMAKE_LFLAGS += '-Wl,-rpath,\'\$${ORIGIN}/lib\''
#/home/daiver/Downloads/Ipopt-3.12.11/build/lib
QMAKE_LFLAGS += '-Wl,-rpath,\'/home/daiver/Downloads/Ipopt-3.12.11/build/lib\''


SOURCES += \
        main.cpp \
    myproblem1.cpp \
    myproblem2.cpp

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

HEADERS += \
    myproblem1.h \
    myproblem2.h
