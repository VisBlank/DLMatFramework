#ifndef AVGPOOLING_H
#define AVGPOOLING_H
#include "baselayer.h"
using namespace std;

class AvgPooling : public BaseLayer
{
public:
    AvgPooling(const string &name, shared_ptr<BaseLayer> inLayer, int kx, int ky, int stride);

    Tensor<float> ForwardPropagation(const Tensor<float> &input) override;
    LayerGradient<float> BackwardPropagation(const LayerGradient<float> &dout) override;
private:
    int m_H_prime, m_W_prime, m_C, m_F, m_HH, m_WW, m_stride;
};


#endif // AVGPOOLING_H
