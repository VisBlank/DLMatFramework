#ifndef DROPOUT_H
#define DROPOUT_H
#include "baselayer.h"

class DropOut : public BaseLayer
{
public:
    DropOut(const string &name, shared_ptr<BaseLayer> inLayer, float dropoutProb);
    Tensor<float> ForwardPropagation(const Tensor<float> &input) override;
    LayerGradient<float> BackwardPropagation(const LayerGradient<float> &dout) override;
    Tensor<float> GetDropoutMask() const;

private:
    Tensor<float> m_dropoutMask;
    float m_dropoutProb;    
};






#endif // DROPOUT_H
