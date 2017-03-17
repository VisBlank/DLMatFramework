#ifndef MEANSQUAREDERROR_H
#define MEANSQUAREDERROR_H
#include "baseloss.h"

class MeanSquaredError : public BaseLoss
{
public:
    MeanSquaredError();
    tuple <float, Tensor<float>> GetLossAndGradients(const Tensor<float> &scores, const Tensor<float> &targets) override;
};

#endif // MEANSQUAREDERROR_H
