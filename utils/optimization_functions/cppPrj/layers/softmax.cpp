#include "softmax.h"
#include "utils/mathhelper.h"

SoftMax::SoftMax(const string &name, shared_ptr<BaseLayer> inLayer){
    m_inputLayer = inLayer;
    m_name = name;
}

Tensor<float> SoftMax::ForwardPropagation(const Tensor<float> &input){
    Tensor<float> activation;
    activation.SetDims(input.GetDims());

    // Fix numerical error
    auto scoresFix = input - (MathHelper<float>::MaxTensor(input,1).first).Repmat(1,input.GetDims()[1]);

    // Get the sum of all scores
    auto sumProb = MathHelper<float>::Sum(MathHelper<float>::Exp(scoresFix),1);

    // Repeat this value for every column of scores
    sumProb = sumProb.Repmat(1,input.GetDims()[1]);

    // Calculate probabilities
    activation = (MathHelper<float>::Exp(scoresFix)).EltWiseDiv(sumProb);

    return activation;
}

LayerGradient<float> SoftMax::BackwardPropagation(const LayerGradient<float> &dout){
    LayerGradient<float> gradient;
    return gradient;
}
