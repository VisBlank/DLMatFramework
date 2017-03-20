/*
Main file

To compile:
g++ -std=c++11 main.cpp -o cppApp

Some references
http://supercomputingblog.com/openmp/tutorial-parallel-for-loops-with-openmp/
http://stackoverflow.com/questions/8384234/return-reference-to-a-vector-member-variable
https://mbevin.wordpress.com/2012/11/20/move-semantics/
*/
#include "utils/tensor.h"
#include "layers/layercontainer.h"
#include "loss/lossfactory.h"
#include "classifier/deeplearningmodel.h"
#include "utils/mathhelper.h"
#include "solver/solver.h"
#include "solver/sgd.h"
#include "layers/baselayer.h"
#include "layers/sigmoid.h"


int main() {    
    Tensor<float> A(vector<int>({2,2}));
    A(0,0) = 1;
    A(0,1) = 2;
    A(1,0) = 3;
    A(1,1) = 4;
    Tensor<float> B(vector<int>({2,2}));
    B(0,0) = 5;
    B(0,1) = 6;
    B(1,0) = 7;
    B(1,1) = 8;

    cout << "Matrix A:" << endl;
    A.print();
    cout << "Matrix B:" << endl;
    B.print();

    Tensor<float>C = A*B; // C.assign(A.mult(B));
    cout << "A*B" << endl;
    C.print();

    Tensor<float>D = A+B;
    cout << "A+B" << endl;
    D.print();

    Tensor<float>E = A-B;
    cout << "A-B" << endl;
    E.print();

    Tensor<float>F = B;
    cout << "F=B" << endl;
    F.print();

    Tensor<float>G = A*(float)2.5;
    cout << "G=A*2.5" << endl;
    G.print();

    Tensor<float>H = A+(float)1.1;
    cout << "H=A*1.1" << endl;
    H.print();

    Tensor<float>I = A.EltWiseMult(A);
    cout << "I=A.*A" << endl;
    I.print();

    Tensor<float>J = A.EltWiseDiv(A);
    cout << "J=A./A" << endl;
    J.print();

    Tensor<float>L = (float)1.0+A;
    cout << "L=1+A" << endl;
    L.print();

    Tensor<float>M = -A;
    cout << "M=-A" << endl;
    M.print();

    cout << "Test SumVec and ProdVec" << endl;
    Tensor<float> someVec(vector<int>({1,4}),{1,2,3,4});
    someVec.print();
    float testSum = MathHelper<float>::SumVec(someVec);
    float testProd = MathHelper<float>::ProdVec(someVec);
    Tensor<float> testLog = MathHelper<float>::Log(someVec);
    cout << "Sum vector someVec=" << testSum << endl;
    cout << "Prod vector someVec=" << testProd << endl;
    testLog.print();

    Tensor<float> input(vector<int>({1,2}),{1.5172, -0.0332});
    Sigmoid sigm("Test",nullptr);
    Tensor<float> fpAct = sigm.ForwardPropagation(input);
    cout << "Sigmoid Forward propagation: ";fpAct.print();
    Tensor<float> dout(vector<int>({1,2}),{-0.3002, 0.2004});
    LayerGradient<float> bpAct = sigm.BackwardPropagation(dout);
    cout << "Sigmoid Backward propagation: ";bpAct.dx.print();


    /*
        Xor problem
    */
    cout << "XOR Problem" << endl;
    // Define input/label matrices(2d tensor)
    Tensor<float> X(vector<int>({4,2}),{0,0,0,1,1,0,1,1});
    Tensor<float> Y(vector<int>({4,1}),{0,1,1,0});
    Tensor<float> Xt(vector<int>({4,2}),{0,0,0,1,1,0,1,1});
    Tensor<float> Yt(vector<int>({4,1}),{0,1,1,0});

    cout << "Xor input" << endl;X.print();
    cout << "Xor output" << endl;Y.print();

    // Define model structure
    LayerContainer layers;
    layers <= LayerMetaData{"Input",LayerType::TInput};
    layers <= LayerMetaData{"FC_1",LayerType::TFullyConnected};
    layers <= LayerMetaData{"Relu_1",LayerType::TSigmoid};
    layers <= LayerMetaData{"FC_2",LayerType::TFullyConnected};
    layers <= LayerMetaData{"Softmax",LayerType::TSoftMax};

    DeepLearningModel net(layers,LossFactory<CrossEntropy>::GetLoss());

    // Create solver and train
    Solver solver(net,OptimizerType::T_SGD, map<string,float>{{"learning_rate",0.1},{"L2_reg",0}});
    solver.SetBatchSize(1);
    solver.SetEpochs(1000);
    solver.Train();

    return 0;
}
