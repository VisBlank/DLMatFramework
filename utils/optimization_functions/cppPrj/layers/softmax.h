#ifndef SOFTMAX_H
#define SOFTMAX_H
#include "baselayer.h"
using namespace std;

class SoftMax : public BaseLayer
{
public:
    SoftMax(const string &name, shared_ptr<BaseLayer> inLayer);
    Tensor<float> ForwardPropagation(const Tensor<float> &input) override;
    LayerGradient<float> BackwardPropagation(const LayerGradient<float> &dout) override;
};
#endif // SOFTMAX_H
