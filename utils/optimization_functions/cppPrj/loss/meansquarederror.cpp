#include "meansquarederror.h"

MeanSquaredError::MeanSquaredError()
{

}

tuple<float, Tensor<float> > MeanSquaredError::GetLossAndGradients(const Tensor<float> &scores, const Tensor<float> &targets)
{
    cout << "From MeanSquaredError" << endl;
    return make_tuple(0.1F,Tensor<float>());
}
