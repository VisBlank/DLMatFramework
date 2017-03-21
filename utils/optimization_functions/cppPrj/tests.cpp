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
#include "loss/crossentropy.h"

TEST_CASE( "Tensor tests "){

    SECTION( "Tensor assignment"){
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
        cout << "Matrix A:" << A << endl;
        cout << "Matrix B:" << B << endl;
    }

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

    SECTION( " Tensor equality " ){
        REQUIRE(A == A);
    }

    SECTION( " Tensor Multiplication " ){
        Tensor<float> D(vector<int>({2,2}));
        D(0,0) = 19;
        D(0,1) = 22;
        D(1,0) = 43;
        D(1,1) = 50;

        Tensor<float>C = A*B; // C.assign(A.mult(B));
        cout << "A*B" << C << endl;
        REQUIRE(D == C);
    }

    SECTION( " Tensor Addition " ){
        Tensor<float> F(vector<int>({2,2}));
        F(0,0) = 6;
        F(0,1) = 8;
        F(1,0) = 10;
        F(1,1) = 12;

        Tensor<float>E = A+B;
        cout << "A+B" << E << endl;
        REQUIRE(E == F);
    }

    SECTION( " Tensor Subtraction " ){
        Tensor<float> H(vector<int>({2,2}));
        H(0,0) = -4;
        H(0,1) = -4;
        H(1,0) = -4;
        H(1,1) = -4;

        Tensor<float>G = A-B;
        cout << "A-B" << G << endl;
        REQUIRE(G == H);
    }

    SECTION( " Tensor assignment " ){

        Tensor<float>I = B;
        cout << "A-B" << I << endl;
        REQUIRE(I == B);
    }

    SECTION( " Tensor scalar multiplication " ){
        Tensor<float> J(vector<int>({2,2}));
        J(0,0) = 2.5;
        J(0,1) = 5;
        J(1,0) = 7.5;
        J(1,1) = 10;

        Tensor<float>K = A*(float)2.5;
        cout << "A*2.5" << K << endl;
        REQUIRE(K == J);
    }

    SECTION( " Tensor scalar addition " ){
        Tensor<float> L(vector<int>({2,2}));
        L(0,0) = 2.1;
        L(0,1) = 3.1;
        L(1,0) = 4.1;
        L(1,1) = 5.1;

        Tensor<float>M = A+(float)1.1;
        cout << "A+1.1" << M << endl;
        REQUIRE(L == M);
    }

    SECTION( " Tensor scalar addition (friend class) " ){
        Tensor<float> L(vector<int>({2,2}));
        L(0,0) = 2.1;
        L(0,1) = 3.1;
        L(1,0) = 4.1;
        L(1,1) = 5.1;

        Tensor<float>M = (float)1.1 + A;
        cout << "A+1.1" << M << endl;
        REQUIRE(L == M);
    }

    SECTION( " Tensor eltwise mult " ){
        Tensor<float> N(vector<int>({2,2}));
        N(0,0) = 1;
        N(0,1) = 4;
        N(1,0) = 9;
        N(1,1) = 16;

        Tensor<float>O = A.EltWiseMult(A);
        cout << "A.*A" << O << endl;
        REQUIRE(N == O);
    }

    SECTION( " Tensor eltwise div " ){
        Tensor<float> P(vector<int>({2,2}));
        P(0,0) = 1;
        P(0,1) = 1;
        P(1,0) = 1;
        P(1,1) = 1;

        Tensor<float>Q = A.EltWiseDiv(A);
        cout << "A./A" << Q << endl;
        REQUIRE(P == Q);
    }

    SECTION( " Tensor negation " ){
        Tensor<float> R(vector<int>({2,2}));
        R(0,0) = -1;
        R(0,1) = -2;
        R(1,0) = -3;
        R(1,1) = -4;

        Tensor<float>S = -A;
        cout << "A./A" << S << endl;
        REQUIRE(R == S);
    }

    SECTION( " Sum(Vec),Prod(Vec) and Log(Vec) function "){

        cout << "Test SumVec and ProdVec" << endl;
        Tensor<float> someVec(vector<int>({1,4}),{1,2,3,4});
        cout << someVec;
        float testSum = MathHelper<float>::SumVec(someVec);
        float testProd = MathHelper<float>::ProdVec(someVec);
        Tensor<float> testLog = MathHelper<float>::Log(someVec);
        cout << "Sum vector someVec=" << testSum << endl;
        REQUIRE( testSum == 10 );
        cout << "Prod vector someVec=" << testProd << endl;
        REQUIRE( testProd == 24 );
        cout << testLog;
        Tensor<float> logSomeVec (vector<int>({1,4}),{0,0.693147,1.09861,1.38629});
        cout << "Difference for log is:" << MathHelper<float>::SumVec(testLog - logSomeVec) << endl;
        REQUIRE( MathHelper<float>::SumVec(testLog - logSomeVec) < 0.01 );

    }

    SECTION( " matrix of zero creation "){

        Tensor<float> zeroMatA(vector<int>({2,4}),{0,0,0,0,0,0,0,0});

        Tensor<float> zerosMat2d = MathHelper<float>::Zeros(vector<int>({2,4}));
        cout << "Zeros 2d[2x4] matrix" << zerosMat2d << endl;

        REQUIRE( zeroMatA == zerosMat2d );

    }

    SECTION( " matrix of random numbers " ){

        Tensor<float> randnMat2d = MathHelper<float>::Randn(vector<int>({4,4}));
        cout << "Normal distribution 2d[4x4] matrix" << randnMat2d << endl;

    }

    SECTION( " sigmoid test " ){

        Tensor<float> input(vector<int>({1,2}),{1.5172, -0.0332});
        Sigmoid sigm("Test",nullptr);
        Tensor<float> fpAct = sigm.ForwardPropagation(input);
        cout << "Sigmoid Forward propagation: " << fpAct << endl;
        Tensor<float> dout(vector<int>({1,2}),{-0.3002, 0.2004});
        LayerGradient<float> bpAct = sigm.BackwardPropagation(dout);
        cout << "Sigmoid Backward propagation: " << bpAct.dx << endl;
    }

    SECTION ( " cross entropy test " ){

        // Test Cross entropy
        Tensor<float> scores(vector<int>({2,1}),{0.3897, 0.4049});
        Tensor<float> targets(vector<int>({2,1}),{0.0, 0.0});
        CrossEntropy cross;
        auto LossGrad = cross.GetLossAndGradients(scores,targets);
        cout << "Loss: " << get<0>(LossGrad) << endl;
        cout << get<1>(LossGrad) << endl;

    }

    SECTION( " Xor problem" ){

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
