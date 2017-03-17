#ifndef MATHHELPER_H
#define MATHHELPER_H

#include <numeric>
#include "tensor.h"
using namespace std;

template<typename T>
class MathHelper
{
public:
    static T SumVec(const Tensor<T> &in);
    static T ProdVec(const Tensor<T> &in);

};

#endif // MATHHELPER_H
