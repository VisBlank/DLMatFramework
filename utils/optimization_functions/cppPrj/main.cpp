/*
Main file

To compile:
g++ -std=c++11 main.cpp -o cppApp

Some references
http://supercomputingblog.com/openmp/tutorial-parallel-for-loops-with-openmp/
http://stackoverflow.com/questions/8384234/return-reference-to-a-vector-member-variable
https://mbevin.wordpress.com/2012/11/20/move-semantics/
*/

#define CATCH_CONFIG_RUNNER
#include "test/catch.hpp"
#define TEST true


#include "utils/tensor.h"
#include "utils/dataset.h"
#include "layers/layercontainer.h"
#include "loss/lossfactory.h"
#include "classifier/deeplearningmodel.h"
#include "utils/mathhelper.h"
#include "solver/solver.h"
#include "solver/sgd.h"
#include "layers/baselayer.h"
#include "layers/sigmoid.h"
#include "loss/crossentropy.h"

int runCatchTests()
{
    return Catch::Session().run();
}

int main() {        
    if (TEST)
    {
        return runCatchTests();
    }       


    Tensor<float> zerosMat2d = MathHelper<float>::Zeros(vector<int>({2,4}));
    cout << "Zeros 2d[2x4] matrix" << zerosMat2d << endl;
    Tensor<float> randnMat2d = MathHelper<float>::Randn(vector<int>({4,4}));
    cout << "Normal distribution 2d[4x4] matrix" << randnMat2d << endl;

    // Test Sigmoid
    Tensor<float> input(vector<int>({1,2}),{1.5172, -0.0332});
    Sigmoid sigm("Test",nullptr);
    Tensor<float> fpAct = sigm.ForwardPropagation(input);
    cout << "Sigmoid Forward propagation: " << fpAct << endl;
    Tensor<float> dout(vector<int>({1,2}),{-0.3002, 0.2004});
    LayerGradient<float> bpAct = sigm.BackwardPropagation(dout);
    cout << "Sigmoid Backward propagation: " << bpAct.dx << endl;

    // Test Cross entropy
    Tensor<float> scores(vector<int>({2,1}),{0.3897, 0.4049});
    Tensor<float> targets(vector<int>({2,1}),{0.0, 0.0});
    CrossEntropy cross;
    auto LossGrad = cross.GetLossAndGradients(scores,targets);
    cout << "Loss: " << get<0>(LossGrad) << endl;
    cout << get<1>(LossGrad) << endl;


    /*
        Xor problem
    */
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
    layers <= LayerMetaData{"Relu_1",LayerType::TSigmoid};
    layers <= LayerMetaData{"FC_2",LayerType::TFullyConnected,1};
    layers <= LayerMetaData{"Softmax",LayerType::TSoftMax};


    DeepLearningModel net(layers,LossFactory<CrossEntropy>::GetLoss());

    // Create solver and train
    Solver solver(net,data,OptimizerType::T_SGD, map<string,float>{{"learning_rate",0.1},{"L2_reg",0}});
    solver.SetBatchSize(1);
    solver.SetEpochs(1000);
    solver.Train();
    auto lossHistory = solver.GetLossHistory();
    lossHistory[2] = 1;

    auto score0 = net.Predict(Tensor<float>(vector<int>({1,2}),{0,0}));
    auto score1 = net.Predict(Tensor<float>(vector<int>({1,2}),{0,1}));
    auto score2 = net.Predict(Tensor<float>(vector<int>({1,2}),{1,0}));
    auto score3 = net.Predict(Tensor<float>(vector<int>({1,2}),{1,1}));
    cout << "0 XOR 0 :" << endl;
    cout << "0 XOR 1 :" << endl;
    cout << "1 XOR 0 :" << endl;
    cout << "1 XOR 1 :" << endl;


    return 0;
}
