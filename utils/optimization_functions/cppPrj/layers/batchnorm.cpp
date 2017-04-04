#include "batchnorm.h"

BatchNorm::BatchNorm(const string &name, shared_ptr<BaseLayer> inLayer){
    m_name = name;
    m_inputLayer = inLayer;
    m_isTraining = true;
}

Tensor<float> BatchNorm::ForwardPropagation(const Tensor<float> &input){

}

LayerGradient<float> BatchNorm::BackwardPropagation(const LayerGradient<float> &dout){

}
