#include "convolution.h"

Convolution::Convolution(const string &name, shared_ptr<BaseLayer> inLayer, int kx, int ky, int stride, int pad, int F){
    m_inputLayer = inLayer;
    m_name = name;
    m_hasParameter = true;

    if (m_inputLayer != nullptr){
        auto shapeInputLayer = m_inputLayer->GetActivationShape();
        auto fan_in = accumulate(shapeInputLayer.begin(), shapeInputLayer.end(),1,multiplies<int>());
        auto C = shapeInputLayer[2];

        m_H_prime = (shapeInputLayer[0]+2*pad-ky)/stride +1;
        m_W_prime = (shapeInputLayer[0]+2*pad-kx)/stride +1;
        m_C = C;
        m_F = F;
        m_HH = ky;
        m_WW = kx;
        m_stride = stride;
        m_pad = pad;

        m_activationShape.push_back(m_H_prime);
        m_activationShape.push_back(m_W_prime);
        m_activationShape.push_back(F);
        m_activationShape.push_back(-1);

        // Initialize weights and bias
        m_weights = MathHelper<float>::Randn(vector<int>({ky,kx,C,F})) / std::sqrt(fan_in);
        m_bias = MathHelper<float>::Zeros(vector<int>({F,1}));

        // Prepare weights and bias for matrix multiplication
        m_weights.Reshape(vector<int>{ky*kx*C, F});
        m_weights = m_weights.Transpose();
    }
}

Tensor<float> Convolution::ForwardPropagation(const Tensor<float> &input){
    auto H = input.GetRows();
    auto W = input.GetCols();
    auto N = input.GetBatch();

    // Alocate memory for output
    Tensor<float> activation(vector<int>({m_H_prime,m_W_prime,m_F,N}));

    // Preparing filter weights
    //auto filter_col = m_weights.Reshape();
    //auto filter_col_T = filter_col.Transpose();

    /*
     * Here we convolve each image on the batch in a for-loop, but the im2col
     *  could also handle a image batch at the input, so all computations would
     *  be just one big matrix multiplication. We opted now for this to test the
     *  par-for implementation with OpenMP on CPU
     *
    */
    for (auto idxBatch = 0; idxBatch < N; ++idxBatch){
        // Select image from batch
        auto img = Tensor<float>::GetTensorFromBatch(input, idxBatch);

        // Convert image to cols
        auto im_col = Tensor<float>::im2col(img,m_HH,m_WW,m_stride,m_pad);

        // Convert to 2d matrix (im2col)
        auto mul = (m_weights * im_col) + m_bias;

        // Set activation
        //activations(:,:,:,idxBatch) =  reshape_row_major_custom(mul,[H_prime W_prime size(mul,1)]);
        //activations.PutTensorOnBatch(ImgB0,idxBatch);
    }

    // Cache results for backpropagation
    m_activation = activation;
    m_previousInput = input;

    return activation;
}

LayerGradient<float> Convolution::BackwardPropagation(const LayerGradient<float> &dout){

}

void Convolution::setWeights(Tensor<float> &weights){
    m_weights = weights;
}

void Convolution::setBias(Tensor<float> &bias){
    m_bias = bias;
}
