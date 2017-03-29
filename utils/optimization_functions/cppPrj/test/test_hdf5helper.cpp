#include "test/catch.hpp"
#include "utils/tensor.h"
#include "layers/layercontainer.h"
#include "loss/lossfactory.h"
#include "classifier/deeplearningmodel.h"
#include "utils/mathhelper.h"
#include "solver/solver.h"
#include "solver/sgd.h"
#include "layers/baselayer.h"
#include "layers/sigmoid.h"
#include "layers/relu.h"
#include "loss/crossentropy.h"
#include "utils/reverse_range_based.h"
#include "utils/range.h"
#include "layers/softmax.h"
#include <array>
#include "utils/hdf5tensor.h"

TEST_CASE( "HDF5 tests"){
    SECTION( "Read dataset" ){
        HDF5Tensor obj(string("/home/leo/work/DLMatFramework/learn/python_notebooks/for_leo.h5"));
        auto someTensor = obj.GetData(string("X"));

        cout << "Tensor:" << someTensor << endl;
    }
}
