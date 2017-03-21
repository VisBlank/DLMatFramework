/*
    Helper for Math functions on tensors
    References:
    http://en.cppreference.com/w/cpp/algorithm/generate
    http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
*/
#ifndef MATHHELPER_H
#define MATHHELPER_H

#include <numeric>
#include <random>
#include <cmath>
#include "tensor.h"
#include <math.h>
using namespace std;

template<typename T>
class MathHelper
{    
public:
    static T SumVec(const Tensor<T> &in);
    static T ProdVec(const Tensor<T> &in);    
    static Tensor<T> Log(const Tensor<T> &in);
    static Tensor<T> Exp(const Tensor<T> &in);
    static Tensor<T> Zeros(const vector<int> &dims);
    static Tensor<T> Randn(const vector<int> &dims);

};

#endif // MATHHELPER_H
