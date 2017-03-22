#include "mathhelper.h"
/*
    References:
    http://stackoverflow.com/questions/115703/storing-c-template-function-definitions-in-a-cpp-file
    http://en.cppreference.com/w/cpp/algorithm/accumulate
*/

template<typename T>
pair<T, unsigned int> MathHelper<T>::MaxVec(const Tensor<T> &in){
    // max_element return an iterator for the biggest value
    auto itMax = max_element(in.begin(),in.end());
    // Fetch the value
    T val = *itMax;
    // Convert the iterator to index
    unsigned int idx = distance(in.begin(), itMax);
    // Return the pair (value,
    return make_pair(val, idx);
}

template<typename T>
Tensor<T> MathHelper<T>::MaxVec(const Tensor<T> &in, const T &scalar){
    Tensor<T> result(vector<int>({in.GetDims()}));
    // Return the biggest element between scalar and in(each element)
    transform(in.begin(), in.end(), result.begin(),[&scalar](T m) -> T {return (m>scalar)?m:scalar;});
    return result;
}

template<typename T>
T MathHelper<T>::SumVec(const Tensor<T> &in){
    T res = accumulate(in.begin(), in.end(),T(0));
    return res;
}

template<typename T>
T MathHelper<T>::ProdVec(const Tensor<T> &in){    
    T res = accumulate(in.begin(), in.end(),1,multiplies<T>());
    return res;
}

template<typename T>
Tensor<T> MathHelper<T>::Log(const Tensor<T> &in){
    Tensor<T> result(vector<int>({in.GetDims()}));    

    // For each element of invVec apply log(element) and store the result on resVec
    transform(in.begin(), in.end(), result.begin(),[](T m) -> T {return log(m);});
    return result;
}

template<typename T>
Tensor<T> MathHelper<T>::Exp(const Tensor<T> &in){
    Tensor<T> result(vector<int>({in.GetDims()}));

    // For each element of invVec apply log(element) and store the result on resVec
    transform(in.begin(), in.end(), result.begin(),[](T m) -> T {return exp(m);});
    return result;
}

template<typename T>
Tensor<T> MathHelper<T>::Zeros(const vector<int> &dims){
    Tensor<T> result(dims);
    fill (result.begin(),result.end(),(T)0);
    return result;
}

template<typename T>
Tensor<T> MathHelper<T>::Randn(const vector<int> &dims){
    random_device rd;
    mt19937 gen(rd());
    // Mean 0, standard deviation 1
    std::normal_distribution<> d(0,1);
    Tensor<T> result(dims);
    generate(result.begin(), result.end(), [&d,&gen]{ return d(gen); });
    return result;
}

// Explicit template instantiation
template class MathHelper<float>;
template class MathHelper<double>;
template class MathHelper<int>;
