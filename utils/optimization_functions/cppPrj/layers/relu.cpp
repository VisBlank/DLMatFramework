#include "relu.h"

Relu::Relu(const string &name, shared_ptr<BaseLayer> inLayer)
{
    m_inputLayer = inLayer;
    m_name = name;

    // Relu does not change the shape of it's input
    if (m_inputLayer != nullptr){
        m_activationShape = m_inputLayer->GetActivationShape();
    }
}

Tensor<float> Relu::ForwardPropagation(const Tensor<float> &input){
    Tensor<float> activation = MathHelper<float>::MaxVec(0,input);

    // Cache information for backpropagation
    m_previousInput = input;
    m_activation = activation;

    return activation;
}

LayerGradient<float> Relu::BackwardPropagation(const Tensor<float> &dout){    
    // Backpropagate only the inputs values that have been selected during the FP
    Tensor<float> dx = dout.EltWiseMult((m_previousInput >= (float)0));
    LayerGradient<float> gradient{dx};

    // Cache gradients
    m_gradients = gradient;

    return gradient;
}
