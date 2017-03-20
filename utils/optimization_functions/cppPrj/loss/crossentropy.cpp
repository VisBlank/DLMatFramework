#include "crossentropy.h"

CrossEntropy::CrossEntropy()
{

}

tuple<float, Tensor<float> > CrossEntropy::GetLossAndGradients(const Tensor<float> &scores, const Tensor<float> &targets)
{
    auto N = scores.GetRows();
    cout << "From CrossEntropy" << endl;
    float loss = 0.1F;

    // TODO: Implement the sum over 2d matrices.
    auto test1 = (-targets).EltWiseMult(MathHelper<float>::Log(scores)) - (1.0-targets).EltWiseMult(MathHelper<float>::Log(1.0-scores));

    loss = MathHelper<float>::SumVec(test1) / N;
    // gradients is the derivative of the loss function w.r.t the scores
    Tensor<float> gradients = scores - targets;

    // Return
    return make_tuple(loss,gradients);
}
