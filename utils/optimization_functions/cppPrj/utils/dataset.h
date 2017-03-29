#ifndef DATASET_H
#define DATASET_H
#include <set>
#include <vector>
#include <algorithm>
#include "tensor.h"
#include "hdf5tensor.h"

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
    size_t m_numClasses = 0;



public:
    Dataset() = delete;
    Dataset(const string &hdf5File, bool doOneHot = false);
    Dataset(const Tensor<T> &X, const Tensor<T> &Y, int numSamples, bool doOneHot = false);
    Batch<T> GetBatch(int numBatches);
    int GetTrainSize() const { return m_trainingSize;}
    void ShuffleEveryNIterations(const int &N){m_shuffleTime = N;}
    size_t GetNumClasses() const;
};

#endif // DATASET_H
