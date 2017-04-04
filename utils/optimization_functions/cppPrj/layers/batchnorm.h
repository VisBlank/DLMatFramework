#ifndef BATCHNORM_H
#define BATCHNORM_H
#include "baselayer.h"

class BatchNorm : public BaseLayer
{
public:    
    BatchNorm(const string &name, shared_ptr<BaseLayer> inLayer);
    Tensor<float> ForwardPropagation(const Tensor<float> &input) override;
    LayerGradient<float> BackwardPropagation(const LayerGradient<float> &dout) override;
};

#endif // BATCHNORM_H
