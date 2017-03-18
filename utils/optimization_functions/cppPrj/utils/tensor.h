/*
Tensor type
References:
http://www.learncpp.com/cpp-tutorial/99-overloading-the-parenthesis-operator/
https://leonardoaraujosantos.gitbooks.io/opencl/content/bigger_matrix_multiply_problem.html
http://stackoverflow.com/questions/27873719/c-get-initializer-list-for-constructor-with-other-parameters
*/
#ifndef TENSOR_H
#define TENSOR_H


#include <iostream>
#include <tuple>
#include <vector>
#include <memory>
#include <algorithm>
#include <array>
#include <stdexcept>
#include <initializer_list>

using namespace std;

#define USE_ROW_MAJOR
#ifdef USE_ROW_MAJOR
#define MAT_2D(i, j, width) (((i) * (width)) + (j))
#define MAT_3D(i, j, k, height, width) (((i) * (width)) + (j) + ((k) * (height) * (width)))
#define MAT_4D(i, j, k, l, height, width, depth) (((i) * (width)) + (j) + ((k) * (height) * (width)) + ((l) * (height) * (width) * (depth)))
#else
#define MAT_2D(i, j, height) (((j) * (height)) + (i))
#define MAT_3D(i, j, k, height, width) (((j) * (height)) + (i) + ((k) * (height) * (width)))
#define MAT_4D(i, j, k, l, height, width, depth) (((j) * (height)) + (i) + ((k) * (height) * (width)) + ((l) * (height) * (width) * (depth)))
#endif

template <typename T>
class Tensor {
private:
    //unique_ptr<T[]> m_buffer;
    vector<T> m_buffer;
    vector<int> m_dims;
    int m_num_dims;
    int m_numElements;    
public:
    // Delete default Constructor
    //Tensor() = delete;
    Tensor(){};
    Tensor(const vector<int> &dims, initializer_list<T> list){
        m_buffer = vector<T>(list.size(),0);
        copy(list.begin(), list.end(), m_buffer.begin());
        SetDims(dims);
    }
    Tensor(initializer_list<T> list){
        m_buffer = vector<T>(list.size(),0);
        copy(list.begin(), list.end(), m_buffer.begin());
    }

    void SetDims(const vector<int> &dims){
        m_dims = dims;
        m_num_dims = dims.size();
        // Do a "prod" of all elements on vector
        int prodDims = 1;
        for_each(dims.begin(), dims.end(), [&] (int m){prodDims *= m;});        
        m_numElements = prodDims;
    }

    Tensor (const vector<int> &dims):m_dims(dims){
        m_num_dims = dims.size();
        // Do a "prod" of all elements on vector
        int prodDims = 1;
        for_each(dims.begin(), dims.end(), [&] (int m){prodDims *= m;});
        // Initialize vector with prodDims size and fill with zeros
        m_buffer = vector<T>(prodDims,0);
        m_numElements = prodDims;
    }

    // A const method does not change it's class members
    // The expected format will be rows,cols,channel,batch
    int GetNumDims() const {return m_num_dims;}
    vector<int> GetDims() const {return m_dims;}
    int GetRows() const {return m_dims[0];}
    int GetCols() const {return m_dims[1];}

    // Return a copy of our buffer (safer)
    vector<T> GetBufferCopy() const {return m_buffer;}

    typename std::vector<T>::iterator begin(){
        return m_buffer.begin();
    }
    typename std::vector<T>::iterator end() {
        return m_buffer.end();
    }
    typename std::vector<T>::const_iterator begin() const {
        return m_buffer.begin();
    }
    typename std::vector<T>::const_iterator end() const {
        return m_buffer.end();
    }

    // The cont here means that this method will not change the class members    
    void print() const{
        auto start = m_buffer.begin();
        auto ncols = m_dims[1];
        auto nrows = m_dims[0];
        for (int i = 0; i < nrows; ++i){
            // Get a slice from the vector
            vector<T> rowSlice(start, start + m_dims[1]);
            cout << "| ";
            for_each(rowSlice.begin(), rowSlice.end(), [](T m){cout << m << " ";});
            start += ncols;
            cout << "|" << endl;
        }
    }

    /*
    Overload the "()" and "*" operators to make it feel like matlab
  */
    T& operator()(int row, int col){
        // Using at is safer because it checks the boundaries of the vector
        //return m_buffer[MAT_2D(row, col,m_dims[1])];
        return m_buffer.at(MAT_2D(row, col,m_dims[1]));
    }

    Tensor<T> operator*(Tensor &b){
        vector<int> bDims = b.GetDims();
        if (b.GetNumDims() > 2){
            throw invalid_argument("Only 2d matrix multiplication is supported.");
        }
        // Create result matrix
        Tensor<T> result(vector<int>({m_dims[0],bDims[1]}));
        int num_rows_A = m_dims[0];
        int num_cols_A = m_dims[1];
        int num_cols_B = bDims[1];
        for(int i=0; i<num_rows_A; ++i) {
            #pragma omp parallel for
            for (int k=0; k<num_cols_B; ++k) {
                T sum = 0;
                for (int j=0; j<num_cols_A; ++j){
                    sum += (*this)(i,j) * b(j,k);
                }
                result(i,k) = sum;
            }
        }
        return result;
    }

    Tensor<T> operator*(const T b) const{
        // Create result tensor with same dimensions
        Tensor<T> result(vector<int>({m_dims}));       

        // For each element of m_buffer multiply by b and store the result on resVec
        transform(m_buffer.begin(), m_buffer.end(), result.begin(),std::bind1st(std::multiplies<T>(),b));

        return result;
    }

    Tensor<T> operator+(const Tensor &b) const {
        // Create result tensor with same dimensions
        Tensor<T> result(vector<int>({m_dims}));
        vector<int> bDims = b.GetDims();
        if (bDims != m_dims){
            throw invalid_argument("Dimensions must match.");
        }        

        // Add contents of A and B and store results on resVec
        transform(m_buffer.begin(), m_buffer.end(), b.begin(),result.begin(), plus<T>());

        return result;
    }

    Tensor<T> operator+(const T b) const {
        // Create result tensor with same dimensions
        Tensor<T> result(vector<int>({m_dims}));

        // For each element of m_buffer multiply by b and store the result on resVec
        transform(m_buffer.begin(), m_buffer.end(), result.begin(),std::bind1st(std::plus<T>(),b));

        return result;
    }

    Tensor<T> operator-(const Tensor &b) const {
        // Create result tensor with same dimensions
        Tensor<T> result(vector<int>({m_dims}));
        vector<int> bDims = b.GetDims();
        if (bDims != m_dims){
            throw invalid_argument("Dimensions must match.");
        }        

        // Add contents of A and B and store results on resVec
        transform(m_buffer.begin(), m_buffer.end(), b.begin(),result.begin(), minus<T>());

        return result;
    }

    Tensor<T> operator=(const Tensor &b){
        // Create result tensor with same dimensions
        Tensor<T> result(vector<int>({m_dims}));        
        copy(b.begin(), b.end(), result.begin());
        return result;
    }

};

#endif // TENSOR_H
