#include "hdf5tensor.h"

HDF5Tensor::HDF5Tensor(const string &filename){
    try{
        // Initialize H5File structure
        m_file = unique_ptr<H5::H5File>(new H5::H5File(filename, H5F_ACC_TRUNC));
    } catch(H5::FileIException error){
        error.printError();
    }
}

Tensor<float> HDF5Tensor::GetData(const string &datasetName){
    Tensor<float> resp;
    try{
        H5::DataSet dataset = m_file->openDataSet(datasetName);
    } catch(H5::DataSetIException error){
        error.printError();
    }

    return resp;
}
