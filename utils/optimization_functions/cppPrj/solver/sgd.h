#ifndef SGD_H
#define SGD_H
#include "baseoptimizer.h"


template <typename T>
class SGD : public BaseOptimizer<T>
{
public:
    SGD();
    tuple<Tensor<T>, Tensor<T>> Optimize(const Tensor<T> &params, const Tensor<T> &grad_params, const Tensor<T> &state) override;
    ~SGD(){
        cout << "SGD destructor" << endl;
    }
};

#endif // SGD_H
