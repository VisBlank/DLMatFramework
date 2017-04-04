TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt
LIBS += -fopenmp -lhdf5 -lhdf5_cpp -lhdf5_hl_cpp
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
    utils/dataset.cpp \
    layers/relu.cpp \
    layers/sigmoid.cpp \    
    layers/fullyconnected.cpp \
    layers/inputlayer.cpp \    
    layers/softmax.cpp \
    test/test_tensors.cpp \
    test/test_optimizers.cpp \
    test/test_xor_example.cpp \
    test/test_layers.cpp \
    test/test_loss.cpp \
    test/test_toy_example.cpp \
    utils/hdf5tensor.cpp \
    test/test_hdf5helper.cpp \
    layers/dropout.cpp

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
    utils/dataset.h \
    utils/reverse_range_based.h \
    utils/range.h \
    utils/hdf5tensor.h \
    layers/dropout.h
