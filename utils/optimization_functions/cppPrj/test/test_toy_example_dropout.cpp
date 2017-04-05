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

TEST_CASE( "Toy example 2 layer FC deep test and Dropout"){

    SECTION("toy example with Dropout"){

        // Create the dataset from a HDF5 file
        Dataset<float> data = Dataset<float>(string("../../../learn/python_notebooks/toyExample.h5"),true);

        // Define model structure
        LayerContainer layers;
        layers <= LayerMetaData{"Input",LayerType::TInput,1,2,1,1};// Rows,Cols,channels,batch-size
        layers <= LayerMetaData{"FC_1",LayerType::TFullyConnected,100};
        layers <= LayerMetaData{"Relu_1",LayerType::TRelu};
        layers <= LayerMetaData{"DRP_1",LayerType::TDropout, (float)0.5};
        layers <= LayerMetaData{"FC_2",LayerType::TFullyConnected,(int)data.GetNumClasses()};
        layers <= LayerMetaData{"Softmax",LayerType::TSoftMax};

        DeepLearningModel net(layers,LossFactory<MultiClassCrossEntropy>::GetLoss());

        // Create solver and train
        Solver solver(net,data,OptimizerType::T_SGD, map<string,float>{{"learning_rate",0.2},{"L2_reg",0}});
        solver.SetBatchSize(300);
        solver.SetEpochs(5000);
        solver.Train();
        auto lossHistory = solver.GetLossHistory();
        auto lastLoss = lossHistory.back();
        REQUIRE( lastLoss < 0.5 );
        // On matlab open like this:
        // loss = h5read('ToyExampleDropout.h5',/lossHistory');
        HDF5Tensor<float>::WriteData("./ToyExampleDropout.h5","data","lossHistory",lossHistory);
    }


}
