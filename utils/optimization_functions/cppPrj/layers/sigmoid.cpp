#include "sigmoid.h"

Sigmoid::Sigmoid(const string &name, shared_ptr<BaseLayer> inLayer){
    m_inputLayer = inLayer;
    m_name = name;

    // Sigmoid does not change the shape of it's input
    if (m_inputLayer != nullptr){
        m_activationShape = m_inputLayer->GetActivationShape();
    }
}

Tensor<float> Sigmoid::ForwardPropagation(const Tensor<float> &input){
    Tensor<float> activation = 1.0/(1.0+MathHelper<float>::Exp(-input));

    // Cache information for backpropagation
    m_previousInput = input;
    m_activation = activation;

    return activation;
}

LayerGradient<float> Sigmoid::BackwardPropagation(const LayerGradient<float> &dout){
    Tensor<float> t = 1.0/(1.0+MathHelper<float>::Exp(-m_previousInput));
    Tensor<float> d_sigm = t.EltWiseMult(1.0-t);
    Tensor<float> dx = dout.dx.EltWiseMult(d_sigm);

    // Create empty tensors
    Tensor<float> dWeights;
    Tensor<float> dBias;
    LayerGradient<float> gradient{dx,dWeights,dBias} ;

    // Cache gradients
    m_gradients = gradient;

    return gradient;
}
