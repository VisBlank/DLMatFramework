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
    Tensor(){}
    Tensor(const vector<int> &dims, initializer_list<T> list);
    // Allows aggregate initialization
    //http://en.cppreference.com/w/cpp/language/aggregate_initialization
    Tensor(initializer_list<T> list);

    Tensor (const vector<int> &dims);

    void SetDims(const vector<int> &dims);

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

    void print() const;

    /*
        Overload the "()" and "*" operators to make it feel like matlab
        The const after the method definition means that this method will not change the class members
    */
    T& operator()(int row, int col);
    Tensor<T> operator*(Tensor &b);
    Tensor<T> operator*(const T b) const;
    Tensor<T> operator+(const Tensor &b) const;
    Tensor<T> operator+(const T b) const;
    Tensor<T> operator-(const Tensor &b) const;
    Tensor<T> operator=(const Tensor &b);

};

#endif // TENSOR_H
