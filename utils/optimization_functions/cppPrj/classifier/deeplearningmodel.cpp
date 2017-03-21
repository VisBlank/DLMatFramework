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

void DeepLearningModel::InitWeights(){    
    // Iterate on all layers (C++11 for each, stuff)
    for (auto layer:m_layers){
        auto currLayer = m_layers(layer);
        auto layerInput = currLayer->GetInputLayer();
        // Skip the input layer
        if (layerInput == nullptr)
            continue;
        auto shapeInput = layerInput->GetActivationShape();
        std::cout << "Layer:" << layer << " input layer:" << layerInput->GetName() << endl;

        // Initialize weights only on layers that need it
        if (typeid(*currLayer) == typeid(FullyConnected)){
            cout << "Create FC" << endl;
        }
    }
}
