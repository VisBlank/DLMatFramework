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
    int m_trainingSize = 0;
    int m_shuffleTime = 0;
    int m_iterationCounter = 0;
    int m_batchPosition = 0;
    vector<int> m_indexShuffle;




public:
    Dataset() = delete;
    Dataset(const Tensor<T> &X, const Tensor<T> &Y, int numSamples, bool doOneHot = false);
    Batch<T> GetBatch(int numBatches);
    int GetTrainSize() const { return m_trainingSize;}
    void ShuffleEveryNIterations(const int &N){m_shuffleTime = N;}
};

#endif // DATASET_H
