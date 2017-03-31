#include "hdf5tensor.h"

template<typename T>
HDF5Tensor<T>::HDF5Tensor(const string &filename){
    try{
        // Initialize H5File structure
        m_file = unique_ptr<H5::H5File>(new H5::H5File(filename, H5F_ACC_RDONLY));
    } catch(H5::FileIException error){
        error.printError();
    }
}

template<typename T>
Tensor<T> HDF5Tensor<T>::GetData(const string &datasetName){
    Tensor<T> resp;
    try{
        H5::DataSet dataset = m_file->openDataSet(datasetName);
        H5::DataSpace dataspace = dataset.getSpace();
        H5T_class_t typeClass = dataset.getTypeClass();

        // get the size of the dataset
        hsize_t rank;
        array<hsize_t, 10> dims_array; dims_array.fill(1);
        rank = dataspace.getSimpleExtentDims(dims_array.data(), NULL); // rank = 1

        auto rows = dims_array[0];
        auto cols = dims_array[1];

        auto numElements = 1;
        for_each(dims_array.begin(), dims_array.end(), [&] (int m){numElements *= m;});

        resp.SetDims(vector<int>({(int)rows,(int)cols}));
        resp.PreAloc();
        //float data_out[numElements];
        unique_ptr<T[]> data_out = unique_ptr<T[]>(new T[numElements]);
        dataset.read(data_out.get(), H5::PredType::NATIVE_FLOAT, dataspace);
        resp.SetDataFromBuffer(std::move(data_out));

        resp = resp.Transpose();

    } catch(H5::DataSetIException error){
        error.printError();
    }

    return resp;
}

template<typename T>
void HDF5Tensor<T>::WriteData(const string &filename, const string &group, const string &datasetName, const Tensor<T> &tensor){
    try {
        H5::H5File file(filename, H5F_ACC_TRUNC);
        int rank = (int) tensor.GetNumDims();
        array<hsize_t, 10> dims_array; dims_array.fill(1);
        dims_array[0] = tensor.GetRows();
        dims_array[1] = tensor.GetCols();
        H5::DataSpace dataspace(rank, dims_array.data());
        vector<T> m_buff(tensor.begin(), tensor.end());

        H5::DataSet dataset = file.createDataSet(datasetName, H5::PredType::NATIVE_FLOAT, dataspace);
        dataset.write(m_buff.data(), H5::PredType::NATIVE_FLOAT, dataspace);

        // Close resources
        dataspace.close();
        dataset.close();
        file.close();
    }
    catch(H5::FileIException error){
        error.printError();
    }
    catch (H5::DataSetIException error){
        error.printError();
    }
}

template<typename T>
void HDF5Tensor<T>::WriteData(const string &filename, const string &group, const string &datasetName, const vector<T> &vec){
    try {
        H5::H5File file(filename, H5F_ACC_TRUNC);
        int rank = (int) 1;
        array<hsize_t, 10> dims_array; dims_array.fill(1);
        dims_array[0] = vec.size();
        H5::DataSpace dataspace(rank, dims_array.data());

        H5::DataSet dataset = file.createDataSet(datasetName, H5::PredType::NATIVE_FLOAT, dataspace);
        dataset.write(vec.data(), H5::PredType::NATIVE_FLOAT, dataspace);

        // Close resources
        dataspace.close();
        dataset.close();
        file.close();
    }
    catch(H5::FileIException error){
        error.printError();
    }
    catch (H5::DataSetIException error){
        error.printError();
    }
}

/*
 * Explicit declare template versions to avoid linker error. (This is needed if we use templates on .cpp files)
*/
template class HDF5Tensor<float>;
template class HDF5Tensor<double>;
template class HDF5Tensor<int>;
