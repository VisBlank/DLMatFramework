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

TEST_CASE( "Tensor tests"){

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

    SECTION( " Tensor scalar addition (friend class)" ){
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
        cout << "Prod vector someVec=" << testProd << endl;
        cout << testLog;

    }


}
