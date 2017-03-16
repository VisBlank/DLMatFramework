TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt
LIBS += -fopenmp
QMAKE_CXXFLAGS += -fopenmp

SOURCES += main.cpp \
    tensor.cpp \
    layercontainer.cpp \
    baseloss.cpp \
    solver.cpp \
    baseoptimizer.cpp \
    deeplearningmodel.cpp \
    sgd.cpp \
    sgdmomentum.cpp \
    adam.cpp \
    multiclasscrossentropy.cpp \
    meansquarederror.cpp \
    crossentropy.cpp \
    lossfactory.cpp \
    relu.cpp

HEADERS += \
    tensor.h \
    baselayer.h \
    layercontainer.h \
    baseloss.h \
    solver.h \
    baseoptimizer.h \
    deeplearningmodel.h \
    sgd.h \
    sgdmomentum.h \
    adam.h \
    multiclasscrossentropy.h \
    meansquarederror.h \
    crossentropy.h \
    lossfactory.h \
    relu.h
