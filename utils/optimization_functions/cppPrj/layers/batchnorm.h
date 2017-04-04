#ifndef BATCHNORM_H
#define BATCHNORM_H
#include "baselayer.h"

class BatchNorm : public BaseLayer
{
public:    
    BatchNorm(const string &name, shared_ptr<BaseLayer> inLayer, float eps, float momentum);
    Tensor<float> ForwardPropagation(const Tensor<float> &input) override;
    LayerGradient<float> BackwardPropagation(const LayerGradient<float> &dout) override;
private:
    float m_eps;
    float m_momentum;
    Tensor<float> m_running_mean;
    Tensor<float> m_running_var;

    Tensor<float> m_xhat;
    Tensor<float> m_xmu;
    Tensor<float> m_sqrtvar;
    Tensor<float> m_var;

};

#endif // BATCHNORM_H
