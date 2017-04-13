#include "maxpooling.h"

MaxPooling::MaxPooling(const string &name, shared_ptr<BaseLayer> inLayer, int kx, int ky, int stride)
{
    m_inputLayer = inLayer;
    m_name = name;
    m_hasParameter = false;

    m_HH = ky;
    m_WW = kx;
    m_stride = stride;
    // Do we need to set anything else up in the constructor?

}

Tensor<float> MaxPooling::ForwardPropagation(const Tensor<float> &input){
    auto H = input.GetRows();
    auto W = input.GetCols();
    auto C = input.GetDepth();
    auto N = input.GetBatch();
    auto H_prime = (H-m_HH)/m_stride +1;
    auto W_prime = (W-m_WW)/m_stride +1;

    // Alocate memory for output
    Tensor<float> activation(vector<int>({H_prime,W_prime,C,N}));

    // Iterate over the output matrix
    for (int depth = 0; depth < C; depth++){

        // TEMPORARY WAY of grabbing the channel slice from our input to peform max pooling on
        // WHEN we can select a channel using select remove. Or maybe i'm stupid and didnt find another way to do this?
        Tensor<float> input_slice(vector<int>({H, W}));
        for (auto rows_sliced = 0; rows_sliced < H; ++rows_sliced){
            for (auto cols_sliced = 0; cols_sliced < W; ++cols_sliced){
                input_slice(rows_sliced,cols_sliced) = input(rows_sliced,cols_sliced,depth);
            }
        }
        // END OF TEMPORARY WAY

        for (int row = 0; row < H_prime; row++){
            for (int col = 0; col <W_prime; col++){

                // Grab an input patch to perform max on (eventually want to be able to use select to get channel as well)
                auto patch = input_slice.Select(range<int>(row*m_HH,row*m_HH+(m_HH-1)),range<int>(col*m_WW,col*m_WW+m_WW-1));

                // Perform MaxTensor to get the row then do MaxVec to get our max element
                //Can MaxTensor return one element rather than a row/col? If so change this line below
                activation(row,col,depth) = MathHelper<float>::MaxVec((MathHelper<float>::MaxTensor(patch,0)).first).first;
            }
        }
    }

    m_activation = activation;
    m_previousInput = input;
    return activation;
}

LayerGradient<float> MaxPooling::BackwardPropagation(const LayerGradient<float> &dout){

}
