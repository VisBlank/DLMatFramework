#include "softmax.h"

SoftMax::SoftMax(const string &name, shared_ptr<BaseLayer> inLayer){
    m_inputLayer = inLayer;
    m_name = name;
}

Tensor<float> SoftMax::ForwardPropagation(const Tensor<float> &input){
    Tensor<float> activation;
    activation.SetDims(input.GetDims());
    return activation;
}

LayerGradient<float> SoftMax::BackwardPropagation(const LayerGradient<float> &dout){
    LayerGradient<float> gradient;
    return gradient;
}
