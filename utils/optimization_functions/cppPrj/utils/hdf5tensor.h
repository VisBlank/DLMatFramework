#ifndef HDF5TENSOR_H
#define HDF5TENSOR_H
#include <iostream>
#include <string>
#include <memory>
#include <H5Cpp.h>

#include "tensor.h"

using namespace std;

class HDF5Tensor
{
private:
    unique_ptr<H5::H5File> m_file;
public:
    HDF5Tensor(const string &filename);
    Tensor<float> GetData(const string &datasetName);
};

#endif // HDF5TENSOR_H
