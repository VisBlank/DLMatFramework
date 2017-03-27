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
#include "loss/multiclasscrossentropy.h"
#include "utils/reverse_range_based.h"
#include "utils/range.h"
#include <array>

TEST_CASE( "Loss tests"){
    SECTION ("Cross entropy test"){
        // Test Cross entropy
        Tensor<float> scores(vector<int>({2,1}),{0.3897, 0.4049});
        Tensor<float> targets(vector<int>({2,1}),{0.0, 0.0});
        CrossEntropy lossFunc;
        auto LossGrad = lossFunc.GetLossAndGradients(scores,targets);
        cout << "Loss: " << get<0>(LossGrad) << endl;
        cout << get<1>(LossGrad) << endl;
    }

    SECTION ("Multi-class Cross entropy test"){
        // Test Cross entropy
        Tensor<float> probs(vector<int>({4,3}),{0.4088, 0.3200, 0.2713, 0.3515, 0.3348, 0.3137, 0.4010, 0.3058, 0.2932, 0.3486, 0.3272, 0.3243});
        Tensor<float> targets(vector<int>({4,3}),{1,0,0,0,1,0,0,1,0,1,0,0});
        float loss_ref = 1.0569;
        Tensor<float> grad_ref(vector<int>({4,3}),{-0.1478, 0.0800, 0.0678, 0.0879, -0.1663, 0.0784, 0.1003, -0.1736, 0.0733, -0.1629, 0.0818, 0.0811});
        cout << grad_ref << endl;

        MultiClassCrossEntropy lossFunc;
        auto LossGrad = lossFunc.GetLossAndGradients(probs,targets);
        cout << "Loss: " << get<0>(LossGrad) << endl;
        auto lossDiff = abs(loss_ref - get<0>(LossGrad));
        REQUIRE( lossDiff < 0.001F );

        auto gradients = get<1>(LossGrad);
        cout << gradients << endl;
        auto gradDiff = MathHelper<float>::SumVec(MathHelper<float>::Abs(gradients - grad_ref));
        cout << gradDiff << endl;
        REQUIRE( gradDiff < 0.001F );
    }
}
