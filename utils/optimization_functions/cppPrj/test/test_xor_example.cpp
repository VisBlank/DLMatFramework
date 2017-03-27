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

        // Fix starting point to compare with different implementations (ie: with Matlab)
        auto fc1 = net.GetLayers()("FC_1");
        auto fc2 = net.GetLayers()("FC_2");
        auto weights_fc1 = fc1->GetWeightsRef();
        auto bias_fc1 = fc1->GetBiasRef();
        auto weights_fc2 = fc2->GetWeightsRef();
        auto bias_fc2 = fc2->GetBiasRef();

        Tensor<float> fc1_weights(vector<int>({2,2}),{0.1709, -0.0224, 0.6261, 0.4194});
        Tensor<float> fc1_bias(vector<int>({1,2}),{0.7202, -0.4302});

        Tensor<float> fc2_weights(vector<int>({2,1}),{-0.7704,0.5143});
        Tensor<float> fc2_bias(vector<int>({1,1}),{-0.0697});

        weights_fc1 = fc1_weights;
        weights_fc2 = fc2_weights;

        // Create solver and train
        Solver solver(net,data,OptimizerType::T_SGD, map<string,float>{{"learning_rate",0.1},{"L2_reg",0}});
        solver.SetBatchSize(1);
        solver.SetEpochs(1000);
        solver.Train();
        auto lossHistory = solver.GetLossHistory();


        auto score0 = net.Predict(Tensor<float>(vector<int>({1,2}),{0,0}))(0);
        auto score1 = net.Predict(Tensor<float>(vector<int>({1,2}),{0,1}))(0);
        auto score2 = net.Predict(Tensor<float>(vector<int>({1,2}),{1,0}))(0);
        auto score3 = net.Predict(Tensor<float>(vector<int>({1,2}),{1,1}))(0);


        /*REQUIRE( score0 == 0 );
        REQUIRE( score1 == 1 );
        REQUIRE( score2 == 1 );
        REQUIRE( score3 == 0 );

        */

        cout << "0 XOR 0 :" << score0 << endl;
        cout << "0 XOR 1 :" << score1 << endl;
        cout << "1 XOR 0 :" << score2 << endl;
        cout << "1 XOR 1 :" << score3 << endl;

    }

}
