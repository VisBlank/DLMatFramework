#ifndef MULTICLASSCROSSENTROPY_H
#define MULTICLASSCROSSENTROPY_H
#include "baseloss.h"

class MultiClassCrossEntropy : public BaseLoss
{
public:
    MultiClassCrossEntropy();
    tuple <float, Tensor<float>> GetLossAndGradients(const Tensor<float> &scores, const Tensor<float> &targets) override;
};

#endif // MULTICLASSCROSSENTROPY_H
