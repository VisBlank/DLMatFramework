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
#include <array>

TEST_CASE("Optimizer tests"){


    SECTION("SGD test"){
        OptimizerState<float> opt;
        map<string, float> conf_test = {{"L2",0.1f},{"learning_rate",0.1f}};
        SGD<float> sgd_test(conf_test);
        Tensor<float> weights_test(vector<int>({2,2}),{0.1709,-0.0224,0.6261,0.4194});
        Tensor<float> dWeights_test(vector<int>({2,2}),{-0.0443,0.0501,-0.0443,0.0501});
        auto testWeights = sgd_test.Optimize(weights_test, dWeights_test, opt);
        Tensor<float> trueWeights(vector<int>({2,2}),{0.1753,-0.0274,0.6305,0.4144});

        cout << "calculated weights: " << trueWeights << endl;
        cout << "true weights: " << testWeights << endl;
        REQUIRE(MathHelper<float>::SumVec( testWeights - trueWeights ) < 0.001);

    }

    SECTION("SGD with momentum test"){

    }

    SECTION("ADAM test"){

    }

}
