#ifndef FULLYCONNECTED_H
#define FULLYCONNECTED_H
#include "baselayer.h"
using namespace std;

class FullyConnected : public BaseLayer
{    
public:
    FullyConnected(const string &name, shared_ptr<BaseLayer> inLayer, int numOutput){
        m_inputLayer = inLayer;
        m_name = name;

        if (m_inputLayer != nullptr){
            auto shapeInputLayer = m_inputLayer->GetActivationShape();
            auto prodInShape = accumulate(shapeInputLayer.begin(), shapeInputLayer.end(),1,multiplies<int>());
            m_activationShape.push_back(prodInShape);
            m_activationShape.push_back(numOutput);

            // Initialize weights and bias
            m_weights = MathHelper<float>::Randn(vector<int>({prodInShape,numOutput}));
            m_bias = MathHelper<float>::Zeros(vector<int>({1,numOutput}));
        }
    }

    Tensor<float> ForwardPropagation(const Tensor<float> &input) override{
        Tensor<float> activation;
        activation.SetDims(input.GetDims());
        return activation;
    }
    LayerGradient<float> BackwardPropagation(const Tensor<float> &dout) override{
        LayerGradient<float> gradient;
        return gradient;
    }
};

#endif // FULLYCONNECTED_H
