#ifndef BASELOSS_H
#define BASELOSS_H
#include <iostream>
#include <tuple>
#include "utils/tensor.h"
using namespace std;

class BaseLoss
{
public:
    virtual tuple <float, Tensor<float>> GetLossAndGradients(const Tensor<float> &scores, const Tensor<float> &targets);
};

#endif // BASELOSS_H
