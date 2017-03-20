#ifndef RELU_H
#define RELU_H
#include "baselayer.h"
using namespace std;

class Relu : public BaseLayer
{
public:
    // We don't pass a shared_ptr or unique_ptr as reference
    Relu(const string &name, shared_ptr<BaseLayer> inLayer){
        m_inputLayer = inLayer;
        m_name = name;
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

#endif // RELU_H
