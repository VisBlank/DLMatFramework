TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt
LIBS += -fopenmp
QMAKE_CXXFLAGS += -fopenmp

SOURCES += main.cpp \    
    solver/adam.cpp \
    solver/sgd.cpp \
    solver/sgdmomentum.cpp \
    solver/solver.cpp \
    loss/baseloss.cpp \
    loss/crossentropy.cpp \
    loss/meansquarederror.cpp \
    loss/multiclasscrossentropy.cpp \
    layers/layercontainer.cpp \
    classifier/deeplearningmodel.cpp \
    utils/tensor.cpp \
    utils/mathhelper.cpp \
    tests.cpp \
    utils/dataset.cpp

HEADERS += \    
    loss/lossfactory.h \
    solver/adam.h \
    solver/sgd.h \
    solver/sgdmomentum.h \
    solver/solver.h \
    loss/baseloss.h \
    loss/crossentropy.h \
    loss/meansquarederror.h \
    loss/multiclasscrossentropy.h \
    solver/baseoptimizer.h \
    layers/baselayer.h \
    layers/layercontainer.h \
    layers/relu.h \
    classifier/deeplearningmodel.h \
    utils/tensor.h \
    test/catch.hpp \
    layers/layermetadata.h \
    layers/fullyconnected.h \
    layers/softmax.h \
    layers/inputlayer.h \
    utils/mathhelper.h \
    layers/sigmoid.h \
    utils/dataset.h
