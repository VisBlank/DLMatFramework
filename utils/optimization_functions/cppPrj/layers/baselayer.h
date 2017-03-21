#ifndef BASELAYER_H
#define BASELAYER_H
#include <string>
#include <memory>
#include <vector>
#include <algorithm>
#include "utils/tensor.h"
#include "utils/mathhelper.h"

using namespace std;

template<typename T>
class LayerGradient {
public:
    Tensor<T> dx;
    Tensor<T> dWeights;
    Tensor<T> dBias;
};

class BaseLayer
{
public:        
    virtual Tensor<float> ForwardPropagation(const Tensor<float> &input) = 0;
    virtual LayerGradient<float> BackwardPropagation(const Tensor<float> &dout) = 0;

    shared_ptr<BaseLayer> GetInputLayer() const { return m_inputLayer;}
    string GetName() const {return m_name;}
    vector<int> GetActivationShape() const {return m_activationShape;}

    Tensor<float> &GetWeightsRef() {return ref(m_weights);}
    Tensor<float> &GetBiasRef() {return ref(m_bias);}

protected:
    // Weights and bias are references, we don't need to store them
    Tensor<float> m_weights;
    Tensor<float> m_bias;

    // We need to cache the activations and gradients for backprop
    Tensor<float> m_activation;
    Tensor<float> m_previousInput;
    Tensor<float> m_gradients;

    vector<int> m_activationShape;

    string m_name;

    // Reference to layers connected to this current layer    
    shared_ptr<BaseLayer> m_inputLayer = nullptr;
};

#endif // BASELAYER_H
