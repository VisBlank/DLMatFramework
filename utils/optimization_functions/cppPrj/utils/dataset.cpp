#include "dataset.h"

template<typename T>
Dataset<T>::Dataset(const Tensor<T> &X, const Tensor<T> &Y, int numSamples, bool doOneHot){
    m_X_Train = X;
    m_Y_Train = X;
    m_trainingSize = numSamples;
    if (doOneHot){
        cout << "Call doOneHot" << endl;
    }
}

template<typename T>
Batch<T> Dataset<T>::GetBatch(int batchSize){
    if ((batchSize <= 0) || (batchSize > m_trainingSize)){
        batchSize = m_trainingSize;
    }
    Batch<T> batch;
    //batch.X = m_X_Train;
    //batch.Y = m_X_Train;
    // TODO: Tensor select
    Tensor<T> Xtest(vector<int>({1,2}),{0,1});
    Tensor<T> Ytest(vector<int>({1,1}),{1});
    batch.X = Xtest;
    batch.Y = Ytest;
    return batch;
}

/*
 * Explicit declare template versions to avoid linker error. (This is needed if we use templates on .cpp files)
*/
template class Dataset<float>;
template class Dataset<double>;
template class Dataset<int>;
