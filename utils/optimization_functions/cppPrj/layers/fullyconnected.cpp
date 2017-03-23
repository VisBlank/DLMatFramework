#include "fullyconnected.h"

FullyConnected::FullyConnected(const string &name, shared_ptr<BaseLayer> inLayer, int numOutput){
    m_inputLayer = inLayer;
    m_name = name;
    m_hasParameter = true;

    if (m_inputLayer != nullptr){
        auto shapeInputLayer = m_inputLayer->GetActivationShape();
        auto prodInShape = accumulate(shapeInputLayer.begin(), shapeInputLayer.end(),1,multiplies<int>());
        m_activationShape.push_back(1);
        m_activationShape.push_back(numOutput);

        // Initialize weights and bias
        m_weights = MathHelper<float>::Randn(vector<int>({prodInShape,numOutput}));
        m_bias = MathHelper<float>::Zeros(vector<int>({1,numOutput}));
    }
}

Tensor<float> FullyConnected::ForwardPropagation (const Tensor<float> &input) {
    //batch size (not fully batch ready)
    int N;

    if (input.GetNumDims() < 3){
        N = input.GetRows();
    }
    else{
        // TODO: Handle batch input
    }
    /*cout << "---------------- FC----------------" << endl;
    cout << "Input" << input;
    cout << "Weights" << m_weights;
    cout << "Bias" << m_bias;*/
    Tensor<float> activation = input*m_weights + (m_bias.Repmat(N,1));

    // Cache results and input for backprop
    m_activation = activation;
    m_previousInput = input;

    return activation;
}

LayerGradient<float> FullyConnected::BackwardPropagation(const Tensor<float> &dout){

    // Recover cache and get its batch size (not fully batch ready)
    if (m_previousInput.GetNumDims() < 3){
        int N = m_previousInput.GetRows();
    }

    Tensor<float> dx = dout * m_weights.Transpose();
    Tensor<float> dWeights = m_previousInput.Transpose() * dout;
    // Bias gradient should have the same shape as the original bias (NOT FINISHED SHOULD BE SUM ACROSS COLS)
    Tensor<float> dBias = (dout);

    LayerGradient<float> gradient{dx,dWeights,dBias} ;

    // cache gradients
    m_gradients = gradient;

    return gradient;
}

void FullyConnected::setWeights(Tensor<float> &weights){
    m_weights = weights;
}

void FullyConnected::setBias(Tensor<float> &bias){
    m_bias = bias;
}
