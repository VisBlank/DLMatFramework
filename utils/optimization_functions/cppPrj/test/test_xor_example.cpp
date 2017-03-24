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

TEST_CASE("Xor problem test"){

    SECTION("Xor problem"){

        cout << "XOR Problem" << endl;
        // Define input/label matrices(2d tensor)
        Tensor<float> X(vector<int>({4,2}),{0,0,0,1,1,0,1,1});
        Tensor<float> Y(vector<int>({4,1}),{0,1,1,0});
        Tensor<float> Xt(vector<int>({4,2}),{0,0,0,1,1,0,1,1});
        Tensor<float> Yt(vector<int>({4,1}),{0,1,1,0});
        Dataset<float> data(X,Y,4);

        cout << "Xor input" << X << endl;
        cout << "Xor output" << Y << endl;

        // Define model structure
        LayerContainer layers;
        layers <= LayerMetaData{"Input",LayerType::TInput,1,2,1,1};// Rows,Cols,channels,batch-size
        layers <= LayerMetaData{"FC_1",LayerType::TFullyConnected,2};
        layers <= LayerMetaData{"Sigm_1",LayerType::TSigmoid};
        layers <= LayerMetaData{"FC_2",LayerType::TFullyConnected,1};
        layers <= LayerMetaData{"Sigm_2",LayerType::TSigmoid};


        DeepLearningModel net(layers,LossFactory<CrossEntropy>::GetLoss());

        // Create solver and train
        Solver solver(net,data,OptimizerType::T_SGD, map<string,float>{{"learning_rate",0.1},{"L2_reg",0}});
        solver.SetBatchSize(1);
        solver.SetEpochs(1000);
        solver.Train();
        auto lossHistory = solver.GetLossHistory();

        // fix predict
        /*
    auto score0 = net.Predict(Tensor<float>(vector<int>({1,2}),{0,0}));
    auto score1 = net.Predict(Tensor<float>(vector<int>({1,2}),{0,1}));
    auto score2 = net.Predict(Tensor<float>(vector<int>({1,2}),{1,0}));
    auto score3 = net.Predict(Tensor<float>(vector<int>({1,2}),{1,1}));


    REQUIRE( score0 == 0 );
    REQUIRE( score1 == 1 );
    REQUIRE( score2 == 1 );
    REQUIRE( score3 == 0 );

    */

        cout << "0 XOR 0 :" << endl;
        cout << "0 XOR 1 :" << endl;
        cout << "1 XOR 0 :" << endl;
        cout << "1 XOR 1 :" << endl;

    }

}
