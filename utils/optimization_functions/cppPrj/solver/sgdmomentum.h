#ifndef SGDMOMENTUM_H
#define SGDMOMENTUM_H
#include "baseoptimizer.h"

template <typename T>
class SGDMomentum : public BaseOptimizer<T>
{
public:
    // Delete the default constructor so it will force everyone to use SGD passing a config
    SGDMomentum() = delete;
    SGDMomentum(const map<string,float> &config);
    Tensor<T> Optimize(const Tensor<T> &params, const Tensor<T> &grad_params, const OptimizerState<T> &state) override;
};

#endif // SGDMOMENTUM_H
