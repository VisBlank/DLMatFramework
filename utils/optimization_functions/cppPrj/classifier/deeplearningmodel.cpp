#include "deeplearningmodel.h"


Tensor<float> DeepLearningModel::Predict(const Tensor<float> &input){
    Tensor<float> A(vector<int>({1,2}),{0,0});
    return A;
}

tuple<float, Tensor<float>> DeepLearningModel::Loss(const Tensor<float> &X, const Tensor<float> &Y){    
    float A = 0.1F;
    Tensor<float> B(vector<int>({1,2}),{0,0});
    return make_tuple(A,B);
}
