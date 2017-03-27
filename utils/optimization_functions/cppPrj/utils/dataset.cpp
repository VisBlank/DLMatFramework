#include "dataset.h"
#include <random>
#include <chrono>
#include "mathhelper.h"
template<typename T>
Dataset<T>::Dataset(const Tensor<T> &X, const Tensor<T> &Y, int numSamples, bool doOneHot){
    m_X_Train = X;
    m_Y_Train = Y;
    m_trainingSize = numSamples;

    for (int ii=0; ii<numSamples; ++ii) m_indexShuffle.push_back(ii);

    if (doOneHot){
        cout << "Call doOneHot" << endl;
    }
}

template<typename T>
Batch<T> Dataset<T>::GetBatch(int batchSize){
    if ((batchSize <= 0) || (batchSize > m_trainingSize)){
        batchSize = m_trainingSize;
    }

    // Shuffle dataset on first iteration or if reaching set iter. Then reset the counter for place in the batch and set iter.
    if (m_iterationCounter == 0 || m_iterationCounter == m_shuffleTime){
        auto seed = std::chrono::system_clock::now().time_since_epoch().count();
        shuffle(m_indexShuffle.begin(), m_indexShuffle.end(),default_random_engine(seed));
        m_batchPosition = 0;
        m_iterationCounter = 0;
    }
    auto selIndex = m_indexShuffle;

    // Check if batch will take more elements than are left in the dataset to take
    if (m_batchPosition + batchSize > selIndex.size()){
        vector<int>::const_iterator first =  selIndex.begin() + m_batchPosition;
        vector<int>::const_iterator last =  selIndex.begin() + m_trainingSize;
        vector<int> LastElementsBatch(first,last);
        int remainingBatchSize = batchSize - LastElementsBatch.size();

        first =  selIndex.begin() ;
        last =  selIndex.begin() + remainingBatchSize ;
        vector<int> FirstElementsBatch(first,last);

        LastElementsBatch.insert(LastElementsBatch.end(),FirstElementsBatch.begin(),FirstElementsBatch.end());
        selIndex = LastElementsBatch;
        m_batchPosition = 0;
    }else{
        vector<int>::const_iterator first =  selIndex.begin() + m_batchPosition;
        vector<int>::const_iterator last =  selIndex.begin() + m_batchPosition + batchSize;
        vector<int> LastElementsBatch(first,last);

        selIndex = LastElementsBatch;
        m_batchPosition += batchSize;
    }

    Batch<T> batch;    
    batch.X = MathHelper<T>::Zeros(vector<int>({selIndex.size(),2}));
    batch.Y = MathHelper<T>::Zeros(vector<int>({selIndex.size(),1}));

    for (int kk = 0; kk < selIndex.size(); ++kk){
        Tensor<T> selected_row = m_X_Train.Select(range<int>(selIndex[kk],selIndex[kk]),range<int>(-1,-1));
        for (int jj = 0; jj < selected_row.GetDims()[1]; ++jj){
            batch.X(kk,jj) = selected_row(0,jj);
        }
        batch.Y(kk,0) = m_Y_Train(selIndex[kk]);
    }
    m_iterationCounter++;

    return batch;
}



/*
 * Explicit declare template versions to avoid linker error. (This is needed if we use templates on .cpp files)
*/
template class Dataset<float>;
template class Dataset<double>;
template class Dataset<int>;
