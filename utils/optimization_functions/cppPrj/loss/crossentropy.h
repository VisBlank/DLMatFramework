/*
    Cross Entropy: Use for binary class classification
    References:
    https://visualstudiomagazine.com/articles/2014/04/01/neural-network-cross-entropy-error.aspx
*/
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
