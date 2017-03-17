#ifndef CROSSENTROPY_H
#define CROSSENTROPY_H
#include "baseloss.h"

class CrossEntropy : public BaseLoss
{
public:
    CrossEntropy();
    tuple <float, Tensor<float>> GetLossAndGradients(const Tensor<float> &scores, const Tensor<float> &targets) override;
};

#endif // CROSSENTROPY_H
