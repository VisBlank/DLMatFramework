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

TEST_CASE( "Layer tests"){

    SECTION( "Sigmoid test" ){

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
        LayerGradient<float> doutGrad = {dout};
        LayerGradient<float> bpAct = sigm.BackwardPropagation(doutGrad);
        cout << "Sigmoid Backward propagation: " << bpAct.dx << endl;
        REQUIRE( MathHelper<float>::SumVec(ref_bp - bpAct.dx) < 0.01 );
    }

    SECTION("ReLu test Forward Propagation"){

        cout << "Relu test Forward Propagation" << endl;
        Tensor<float> input(vector<int>({1,8}),{-1,2,3,-4,5,-6,7,8});
        Tensor<float> ref_fp(vector<int>({1,8}),{0,2,3,0,5,0,7,8});
        Relu relu("Relu1", nullptr);
        Tensor<float> fpAct = relu.ForwardPropagation(input);
        REQUIRE( MathHelper<float>::SumVec(ref_fp - fpAct) < 0.001 );
        cout << "Relu Forward propagation: " << fpAct << endl;
    }

    SECTION("ReLu test Backward Propagation"){

        cout << "Relu test Forward Propagation" << endl;
        Tensor<float> input(vector<int>({3,4}),{0.0784, 0.0117, 0.6391, -1.2539, -0.0050, 0.5033, -0.4471, 1.7654, 0.0938, 0.1949, 0.6188, -0.8953 });
        Tensor<float> ref_fp(vector<int>({3,4}),{0.0784, 0.0117, 0.6391, 0, 0, 0.5033, 0, 1.7654, 0.0938, 0.1949, 0.6188, 0 });
        Relu relu("Relu1", nullptr);
        Tensor<float> fpAct = relu.ForwardPropagation(input);
        REQUIRE( MathHelper<float>::SumVec(ref_fp - fpAct) < 0.001 );
        cout << "Relu Forward propagation: " << fpAct << endl;

        // Backprop
        Tensor<float> dx(vector<int>({3,4}),{0.0643, -0.0081, 0.3181, -0.0099, -0.0417, 0.0324, -0.1325, -0.0783, 0.0595, -0.0019, 0.3098, -0.0267 });
        Tensor<float> dxGradRef(vector<int>({3,4}),{0.0643, -0.0081, 0.3181, 0, 0, 0.0324, 0, -0.0783, 0.0595, -0.0019, 0.3098, 0 });
        Tensor<float> dWeight, dBias;
        LayerGradient<float> doutGrad = {dx,dWeight,dBias};
        LayerGradient<float> bpAct = relu.BackwardPropagation(doutGrad);
        cout << "Relu Backward propagation: " << bpAct.dx << endl;
        REQUIRE( MathHelper<float>::SumVec(dxGradRef - bpAct.dx) < 0.001 );
    }

    SECTION("Fully connected test"){

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


        // Test FC Backward propagation
        Tensor<float> gradOut(vector<int>({1,2}),{0.5,1.2});
        LayerGradient<float> d_grad = {gradOut};
        LayerGradient<float> bpGrad = fc1.BackwardPropagation(d_grad);


        Tensor<float> actual_dBias(vector<int>({1,2}),{0.5,1.2});
        Tensor<float> actual_dWeights(vector<int>({5,2}),{0.5,1.2,1,2.4,1.5,3.6,2,4.8,2.5,6});
        Tensor<float> actual_dx(vector<int>({1,5}),{2.9,2.9,2.9,2.9,2.9});
        cout << "fc Backward propagation dbias: " << bpGrad.dBias << endl;
        cout << "fc Backward propagation dweights: " << bpGrad.dWeights << endl;
        cout << "fc Backward propagation dx: " << bpGrad.dx << endl;
        REQUIRE( bpGrad.dBias == actual_dBias);
        REQUIRE (MathHelper<float>::SumVec( bpGrad.dWeights - actual_dWeights) < 0.001);
        REQUIRE( bpGrad.dx == actual_dx);

    }
}
