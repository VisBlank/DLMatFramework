#ifndef RELU_H
#define RELU_H
#include "baselayer.h"
using namespace std;

class Relu : public BaseLayer
{
public:
    // We don't pass a shared_ptr or unique_ptr as reference
    Relu(const string &name, shared_ptr<BaseLayer> inLayer);
    Tensor<float> ForwardPropagation(const Tensor<float> &input) override;
    LayerGradient<float> BackwardPropagation(const Tensor<float> &dout) override;
};

#endif // RELU_H
