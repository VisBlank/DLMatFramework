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
#include <utility>
using namespace std;

template<typename T>
class MathHelper
{    
public:
    static pair<T,unsigned int> MaxVec(const Tensor<T> &in);
    static Tensor<T> MaxVec(const Tensor<T> &in, const T &scalar);
    static Tensor<T> MaxVec(const T &scalar, const Tensor<T> &in);    
    static pair<Tensor<T>, Tensor<T>> MaxTensor(const Tensor<T> &in, int dim = 0);
    static T SumVec(const Tensor<T> &in);
    static Tensor<T> Sum(const Tensor<T> &in, int dim);
    static T ProdVec(const Tensor<T> &in);    
    static Tensor<T> Log(const Tensor<T> &in);
    static Tensor<T> Exp(const Tensor<T> &in);
    static Tensor<T> Zeros(const vector<int> &dims);
    static Tensor<T> Randn(const vector<int> &dims);    
};

#endif // MATHHELPER_H
