QT -= gui

CONFIG += c++14 console
CONFIG -= app_bundle


INCLUDEPATH += /usr/include/mkl
INCLUDEPATH += /usr/include/eigen3
DEFINES += EIGEN_USE_MKL_ALL
DEFINES += MKL_LP64
LIBS += -L/usr/lib/x86_64-linux-gnu/mkl/ -lmkl_rt
LIBS += -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

SOURCES += \
        main.cpp

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
