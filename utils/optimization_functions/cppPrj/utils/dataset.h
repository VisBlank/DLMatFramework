#ifndef DATASET_H
#define DATASET_H
#include "tensor.h"

using namespace std;

template <typename T>
class Batch{
public:
    Tensor<T> X;
    Tensor<T> Y;
};

template <typename T>
class Dataset
{
private:
    Tensor<T> m_X_Train;
    Tensor<T> m_Y_Train;
    int m_trainingSize;
    int m_shuffleTime;
public:
    Dataset() = delete;
    Dataset(const Tensor<T> &X, const Tensor<T> &Y, int numSamples, bool doOneHot = false);
    Batch<T> GetBatch(int numBatches);
    int GetTrainSize() const { return m_trainingSize;}
    void ShuffleEveryNIterations(const int &N){m_shuffleTime = N;}
};

#endif // DATASET_H
