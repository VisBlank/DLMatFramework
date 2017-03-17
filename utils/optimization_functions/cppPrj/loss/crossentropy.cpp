#include "crossentropy.h"

CrossEntropy::CrossEntropy()
{

}

tuple<float, Tensor<float> > CrossEntropy::GetLossAndGradients(const Tensor<float> &scores, const Tensor<float> &targets)
{
    auto N = scores.GetRows();
    cout << "From CrossEntropy" << endl;
    Tensor<float> gradients = scores - targets;
    return make_tuple(0.1F,gradients);
}
