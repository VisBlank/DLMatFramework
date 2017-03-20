#ifndef FULLYCONNECTED_H
#define FULLYCONNECTED_H
#include "baselayer.h"
using namespace std;

class FullyConnected : public BaseLayer
{
public:
    FullyConnected(const string &name, shared_ptr<BaseLayer> inLayer){
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

#endif // FULLYCONNECTED_H
