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
#include <array>

TEST_CASE( "Tensor tests "){

    SECTION("Reverse ranged based for"){
        cout << "Reverse ranged based for" << endl;
        vector<int> my_vector{1, 2, 3, 4};
        vector<int> rev_vector_ref{4, 3, 2, 1};
        vector<int> rev_vector;
        cout << "Vector my_vector: ";
        for_each(my_vector.begin(), my_vector.end(), [](int &n){ cout << n << " "; });
        cout << endl;
        auto posArray = my_vector.size();
        for (auto &c : reverse(my_vector)) {
                cout << "my_array[" << posArray << "]=" << c << endl;
                //c = 2;
                rev_vector.push_back(c);
                posArray++;
        }
        REQUIRE(rev_vector == rev_vector_ref);
    }

    SECTION("Tensor(vec) max"){
        cout << "Tensor(vec) max" << endl;
        Tensor<int> A(vector<int>({1,5}),{1,2,3,49,9});
        cout << "A[1x5]" << A << endl;
        auto argMax = MathHelper<int>::MaxVec(A);
        cout << "max(A)=[" << argMax.first << "," << argMax.second << "]" << endl;
        REQUIRE(argMax.first == 49);
        REQUIRE(argMax.second == 3);
    }

    SECTION("Tensor(vec) max between values"){
        cout << "Tensor(vec) max between values" << endl;
        Tensor<int> A(vector<int>({1,8}),{-1,2,3,-4,5,-6,7,8});
        Tensor<int> A_max_ref(vector<int>({1,8}),{0,2,3,0,5,0,7,8});
        cout << "A[1x8]" << A << endl;
        auto maxVec = MathHelper<int>::MaxVec(A,0);
        cout << "max(A,0)=" << maxVec << endl;
        REQUIRE(maxVec == A_max_ref);
    }

    SECTION("Tensor >= boolean operator"){
        cout << "Tensor >= boolean operator" << endl;
        Tensor<int> A(vector<int>({1,8}),{-1,2,3,-4,5,-6,7,8});
        Tensor<int> A_ref(vector<int>({1,8}),{0,1,1,0,1,0,1,1});
        cout << "A[1x8]" << A << endl;
        auto boolOp = (A >= 0);
        cout << "(A >= 0):" << boolOp << endl;
        REQUIRE(A_ref == boolOp);
    }

    SECTION("Tensor <= boolean operator"){
        cout << "Tensor <= boolean operator" << endl;
        Tensor<int> A(vector<int>({1,8}),{-1,2,3,-4,5,-6,7,8});
        Tensor<int> A_ref(vector<int>({1,8}),{1,0,0,1,0,1,0,0});
        cout << "A[1x8]" << A << endl;
        auto boolOp = (A <= 0);
        cout << "(A <= 0):" << boolOp << endl;
        REQUIRE(A_ref == boolOp);
    }

    SECTION("Tensor == boolean operator"){
        cout << "Tensor == boolean operator" << endl;
        Tensor<int> A(vector<int>({1,8}),{-1,2,3,-4,5,-6,7,8});
        Tensor<int> A_ref(vector<int>({1,8}),{0,1,0,0,0,0,0,0});
        cout << "A[1x8]" << A << endl;
        auto boolOp = (A == 2);
        cout << "(A == 2):" << boolOp << endl;
        REQUIRE(A_ref == boolOp);
    }

    SECTION("Tensor != boolean operator"){
        cout << "Tensor != boolean operator" << endl;
        Tensor<int> A(vector<int>({1,8}),{-1,2,3,-4,5,-6,7,8});
        Tensor<int> A_ref(vector<int>({1,8}),{1,0,1,1,1,1,1,1});
        cout << "A[1x8]" << A << endl;
        auto boolOp = (A != 2);
        cout << "(A != 2):" << boolOp << endl;
        REQUIRE(A_ref == boolOp);
    }

    SECTION("Tensor repmat"){
        cout << "Tensor repmat" << endl;
        Tensor<int> A(vector<int>({1,2}),{1,2});
        Tensor<int> A_rep_ref_1_2(vector<int>({1,4}),{1,2,1,2});
        Tensor<int> A_rep_ref_2_1(vector<int>({2,2}),{1,2,1,2});
        Tensor<int> A_rep_ref_2_2(vector<int>({2,4}),{1,2,1,2,1,2,1,2});
        cout << "A[1x2]" << A << endl;
        cout << "repmat(A,[1,2])" << A.Repmat(1,2) << endl;
        cout << "repmat(A,[2,1])" << A.Repmat(2,1) << endl;
        cout << "repmat(A,[2,2])" << A.Repmat(2,2) << endl;

        REQUIRE(A_rep_ref_1_2 == A.Repmat(1,2));
        REQUIRE(A_rep_ref_2_1 == A.Repmat(2,1));
        REQUIRE(A_rep_ref_2_2 == A.Repmat(2,2));
    }

    SECTION("Tensor reshape"){
        cout << "Tensor reshape" << endl;
        Tensor<float> A(vector<int>({3,4}),{1,2,3,4,5,6,7,8,9,10,11,12});
        cout << "A[3x4]" << A << endl;
        CHECK_NOTHROW(A.Reshape(vector<int>({1,12})));
        cout << "A[1x12]" << A << endl;
        CHECK_NOTHROW(A.Reshape(vector<int>({12,1})));
        cout << "A[12x1]" << A << endl;
        CHECK_NOTHROW(A.Reshape(vector<int>({2,6})));
        cout << "A[2x6]" << A << endl;
        CHECK_NOTHROW(A.Reshape(vector<int>({6,2})));
        cout << "A[6x2]" << A << endl;

        // Should throw exception
        REQUIRE_THROWS( A.Reshape(vector<int>({2,60})));
        REQUIRE_THROWS_WITH( A.Reshape(vector<int>({2,60})), "Number of elements must be the same on new shape." );
    }

    SECTION("2d Matrix transpose"){
        cout << "2d Matrix transpose" << endl;
        Tensor<int> A(vector<int>({3,4}),{1,2,3,4,5,6,7,8,9,10,11,12});
        Tensor<int> A_transp_ref(vector<int>({4,3}),{1,5,9,2,6,10,3,7,11,4,8,12});
        cout << "A[3x4]=" << A;
        Tensor<int> A_transp = A.Transpose();
        cout << "A[4x3]=" << A_transp;
        REQUIRE(A_transp_ref == A_transp);
    }

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
        cout << "Sigmoid test" << endl;
        // Test sigmoid Forward propagation (Matlab toy example)
        Tensor<float> ref_fp(vector<int>({1,2}),{0.8201, 0.4917});
        Tensor<float> input(vector<int>({1,2}),{1.5172, -0.0332});
        Sigmoid sigm("Test",nullptr);
        Tensor<float> fpAct = sigm.ForwardPropagation(input);        
        REQUIRE( MathHelper<float>::SumVec(ref_fp - fpAct) < 0.01 );
        cout << "Sigmoid Forward propagation: " << fpAct << endl;

        // Test sigmoid Backward propagation (Matlab toy example)
        Tensor<float> ref_bp(vector<int>({1,2}),{-0.0443, 0.0501});
        Tensor<float> dout(vector<int>({1,2}),{-0.3002, 0.2004});
        LayerGradient<float> bpAct = sigm.BackwardPropagation(dout);
        cout << "Sigmoid Backward propagation: " << bpAct.dx << endl;
        REQUIRE( MathHelper<float>::SumVec(ref_bp - bpAct.dx) < 0.01 );
    }

    SECTION("ReLu test"){
        cout << "Relu test" << endl;
        Tensor<float> input(vector<int>({1,8}),{-1,2,3,-4,5,-6,7,8});
        Tensor<float> ref_fp(vector<int>({1,8}),{0,2,3,0,5,0,7,8});
        Relu relu("Relu1", nullptr);
        Tensor<float> fpAct = relu.ForwardPropagation(input);
        REQUIRE( MathHelper<float>::SumVec(ref_fp - fpAct) < 0.01 );
        cout << "Relu Forward propagation: " << fpAct << endl;
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

    SECTION( " fully connected test " ){

        // Test FC Forward propagation
        Tensor<float> input_test(vector<int>({1,5}),{1,2,3,4,5});
        Tensor<float> weights_test(vector<int>({5,2}),{1,2,1,2,1,2,1,2,1,2});
        Tensor<float> bias_test(vector<int>({1,2}),{1,1});
        FullyConnected fc1("fc1Test",nullptr,2);
        fc1.setWeights( weights_test);
        fc1.setBias( bias_test);

        Tensor<float> fpAct = fc1.ForwardPropagation(input_test);


        Tensor<float> actual_result(vector<int>({1,2}),{16,31});

        cout << "fc Forward propagation: " << fpAct << endl;
        REQUIRE(fpAct == actual_result);

    }


}
