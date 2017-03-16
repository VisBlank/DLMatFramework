#ifndef RELU_H
#define RELU_H
#include "baselayer.h"
using namespace std;

class Relu : public BaseLayer
{
public:
    Relu(string name, int index);
    Tensor<float> ForwardPropagation(const Tensor<float> &input) override;
    Tensor<float> BackwardPropagation() override;
};

#endif // RELU_H
