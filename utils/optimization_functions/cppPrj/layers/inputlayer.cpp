#include "inputlayer.h"

InputLayer::InputLayer(const string &name, int numRows, int numCols, int numChannels, int batchSize){
    m_inputLayer = nullptr;
    m_activationShape.push_back(numRows);
    m_activationShape.push_back(numCols);
    m_activationShape.push_back(numChannels);
    m_activationShape.push_back(batchSize);
    m_name = name;
}

Tensor<float> InputLayer::ForwardPropagation(const Tensor<float> &input){
    Tensor<float> activation;
    activation.SetDims(input.GetDims());
    return activation;
}

LayerGradient<float> InputLayer::BackwardPropagation(const Tensor<float> &dout){
    LayerGradient<float> gradient;
    return gradient;
}
