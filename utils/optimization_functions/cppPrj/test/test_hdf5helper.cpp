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
        HDF5Tensor<float> obj(string("../../../learn/python_notebooks/for_leo.h5"));
        auto X = obj.GetData(string("X"));
        Tensor<float> X_ref(vector<int>({2,3}),{1,2,3,4,5,6});
        cout << "Tensor X:" << X << endl;
        REQUIRE(X == X_ref);

        auto Y = obj.GetData(string("Y"));
        Tensor<float> Y_ref(vector<int>({4,1}),{5,6,7,8});
        cout << "Tensor Y:" << Y << endl;
        REQUIRE(Y == Y_ref);
    }
}
