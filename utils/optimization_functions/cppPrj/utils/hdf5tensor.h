/*
    HDF5 Helper
    References:
    http://stackoverflow.com/questions/25568446/loading-data-from-hdf5-to-vector-in-c
    http://stackoverflow.com/questions/17110435/reading-hdf5-files-to-dynamic-arrays-in-c
*/
#ifndef HDF5TENSOR_H
#define HDF5TENSOR_H
#include <iostream>
#include <string>
#include <memory>
#include <H5Cpp.h>

#include "tensor.h"

using namespace std;
#define NAME_OF_VARIABLE( v ) #v

template <typename T>
class HDF5Tensor
{
private:
    unique_ptr<H5::H5File> m_file;
public:
    HDF5Tensor(const string &filename);
    Tensor<T> GetData(const string &datasetName);
    static void WriteData(const string &filename, const string &group, const string &datasetName, const Tensor<T> &tensor);
    static void WriteData(const string &filename, const string &group, const string &datasetName, const vector<T> &vec);
};

#endif // HDF5TENSOR_H
