#ifndef FULLYCONNECTED_H
#define FULLYCONNECTED_H
#include "baselayer.h"
using namespace std;

class FullyConnected : public BaseLayer
{    
public:
    FullyConnected(const string &name, shared_ptr<BaseLayer> inLayer, int numOutput);

    Tensor<float> ForwardPropagation(const Tensor<float> &input) override;
    LayerGradient<float> BackwardPropagation(const LayerGradient<float> &dout) override;

    void setWeights(Tensor<float> &weights);
    void setBias(Tensor<float> &bias);

};

#endif // FULLYCONNECTED_H
