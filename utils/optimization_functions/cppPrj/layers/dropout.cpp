#include "dropout.h"

DropOut::DropOut(const string &name, shared_ptr<BaseLayer> inLayer, float prob){
    m_name = name;
    m_inputLayer = inLayer;
    m_dropoutProb = prob;
    m_isTraining = true;

    // Dropout does not change the shape of it's input
    if (m_inputLayer != nullptr){
        m_activationShape = m_inputLayer->GetActivationShape();
    }
}

Tensor<float> DropOut::ForwardPropagation(const Tensor<float> &input){
    Tensor<float> activation;
    if (m_isTraining){
        // Create a random matrix with the format of the input vector
        auto randMatrix = MathHelper<float>::Randn(vector<int>({input.GetRows(),input.GetCols()}));
        m_dropoutMask = (randMatrix >= m_dropoutProb) / (1-m_dropoutProb);
        activation = input.EltWiseMult(m_dropoutMask);
    } else {
        activation = input;
    }

    // Cache information for backpropagation
    m_previousInput = input;
    m_activation = activation;

    return activation;
}

LayerGradient<float> DropOut::BackwardPropagation(const LayerGradient<float> &dout){
    // During the backprop use the same mask from training
    Tensor<float> dx = dout.dx.EltWiseMult(m_dropoutMask);
    LayerGradient<float> gradient{dx};

    // Cache gradients
    m_gradients = gradient;

    return gradient;
}

Tensor<float> DropOut::GetDropoutMask() const{
    return m_dropoutMask;
}
