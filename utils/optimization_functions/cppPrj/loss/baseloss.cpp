#include "baseloss.h"


tuple<float, Tensor<float> > BaseLoss::GetLossAndGradients(const Tensor<float> &scores, const Tensor<float> &targets)
{
    cout << "From BaseLoss" << endl;
    return make_tuple(0.1F,Tensor<float>());
}
