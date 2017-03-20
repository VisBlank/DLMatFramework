/*
    Implement the logistic function
    References:
    https://en.wikipedia.org/wiki/Logistic_function
    https://en.wikipedia.org/wiki/Sigmoid_function
    https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
*/
#ifndef SIGMOID_H
#define SIGMOID_H

#include "baselayer.h"
#include "utils/mathhelper.h"
using namespace std;

class Sigmoid : public BaseLayer
{
public:
    // We don't pass a shared_ptr or unique_ptr as reference
    Sigmoid(const string &name, shared_ptr<BaseLayer> inLayer){
        m_inputLayer = inLayer;
        m_name = name;
    }
    Tensor<float> ForwardPropagation(const Tensor<float> &input) override{        
        Tensor<float> activation = 1.0/(1.0+MathHelper<float>::Exp(-input));

        // Cache information for backpropagation
        m_previousInput = input;
        m_activation = activation;

        return activation;
    }
    LayerGradient<float> BackwardPropagation(const Tensor<float> &dout) override{
        Tensor<float> t = 1.0/(1.0+MathHelper<float>::Exp(-m_previousInput));
        Tensor<float> d_sigm = t.EltWiseMult(1.0-t);
        Tensor<float> dx = dout.EltWiseMult(d_sigm);
        LayerGradient<float> gradient{dx};
        return gradient;
    }
};

#endif // SIGMOID_H
