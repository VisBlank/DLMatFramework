#ifndef INPUTLAYER_H
#define INPUTLAYER_H
#include "baselayer.h"
using namespace std;

class InputLayer : public BaseLayer
{
public:
    InputLayer(const string &name, int numRows, int numCols, int numChannels, int batchSize){
        m_inputLayer = nullptr;
        m_activationShape.push_back(numRows);
        m_activationShape.push_back(numCols);
        m_activationShape.push_back(numChannels);
        m_activationShape.push_back(batchSize);
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
#endif // INPUTLAYER_H
