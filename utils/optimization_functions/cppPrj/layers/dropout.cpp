#include "dropout.h"

DropOut::DropOut(const string &name, shared_ptr<BaseLayer> inLayer, float prob)
{
    m_name = name;
    m_inputLayer = inLayer;
    m_dropoutProb = prob;
}

Tensor<float> DropOut::ForwardPropagation(const Tensor<float> &input)
{


    /*Tensor<float> m_dropoutMask = MathHelper<float>::Randn(vector<int>({input.GetRows(),input.GetCols()}));
    m_dropoutMask = m_dropoutMask > m_dropoutProb;

    activation = input .* m_dropoutMask;

    return activation;*/
}

LayerGradient<float> DropOut::BackwardPropagation(const LayerGradient<float> &dout)
{

    //return gradient;
}
