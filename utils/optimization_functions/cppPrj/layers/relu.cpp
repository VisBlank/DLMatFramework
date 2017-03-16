#include "relu.h"

Relu::Relu(string name, int index)
{
    m_name = name;
    m_index = index;
}

Tensor<float> Relu::ForwardPropagation(const Tensor<float> &input)
{
    Tensor<float> activation;
    activation.SetDims(input.GetDims());
    return activation;
}

Tensor<float> Relu::BackwardPropagation()
{
    Tensor<float> gradient;
    return gradient;
}
