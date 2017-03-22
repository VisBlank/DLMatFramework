/*
    Implement the logistic function
    References:
    https://en.wikipedia.org/wiki/Logistic_function
    https://en.wikipedia.org/wiki/Sigmoid_function
    https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
*/
#ifndef SIGMOID_H
#define SIGMOID_H

#include "baselayer.h"
#include "utils/mathhelper.h"
using namespace std;

class Sigmoid : public BaseLayer
{
public:
    // We don't pass a shared_ptr or unique_ptr as reference
    Sigmoid(const string &name, shared_ptr<BaseLayer> inLayer);
    Tensor<float> ForwardPropagation(const Tensor<float> &input) override;
    LayerGradient<float> BackwardPropagation(const Tensor<float> &dout) override;
};

#endif // SIGMOID_H
