#ifndef SGD_H
#define SGD_H
#include "baseoptimizer.h"
using namespace std;


template <typename T>
class SGD : public BaseOptimizer<T>
{
public:
    SGD();
    Tensor<T> Optimize(const Tensor<T> &params, const Tensor<T> &grad_params, const OptimizerState<T> &state) override;
    ~SGD(){
        cout << "SGD destructor" << endl;
    }
};

#endif // SGD_H
