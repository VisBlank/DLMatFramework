#ifndef CONVOLUTION_H
#define CONVOLUTION_H
#include "baselayer.h"
using namespace std;

class Convolution : public BaseLayer
{
public:
    Convolution(const string &name, shared_ptr<BaseLayer> inLayer, int kx, int ky, int stride, int pad, int F);

    Tensor<float> ForwardPropagation(const Tensor<float> &input) override;
    LayerGradient<float> BackwardPropagation(const LayerGradient<float> &dout) override;

    void setWeights(Tensor<float> &weights);
    void setBias(Tensor<float> &bias);
private:
    int m_H_prime, m_W_prime, m_C, m_F, m_HH, m_WW, m_stride, m_pad;

};

#endif // CONVOLUTION_H
