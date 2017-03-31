/*
Tensor type
References:
http://www.learncpp.com/cpp-tutorial/99-overloading-the-parenthesis-operator/
https://leonardoaraujosantos.gitbooks.io/opencl/content/bigger_matrix_multiply_problem.html
http://stackoverflow.com/questions/27873719/c-get-initializer-list-for-constructor-with-other-parameters
http://www.learncpp.com/cpp-tutorial/92-overloading-the-arithmetic-operators-using-friend-functions/
*/
#ifndef TENSOR_H
#define TENSOR_H


#include <iostream>
#include <tuple>
#include <vector>
#include <array>
#include <memory>
#include <algorithm>
#include <array>
#include <stdexcept>
#include <initializer_list>
#include <sstream>
#include "utils/range.h"

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
    int m_num_dims = 0;
    int m_numElements = 0;
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
    void PreAloc() {m_buffer = vector<T>(m_numElements,0);}
    void Reshape(const vector<int> &newdims);
    void SetDataFromBuffer(unique_ptr<T[]> ptrBuff);
    void SaveToHDF5(const string &fileName) const;

    // The expected format will be rows,cols,channel,batch
    int GetNumDims() const {return m_num_dims;}
    vector<int> GetDims() const {return m_dims;}
    int GetRows() const {return m_dims[0];}
    int GetCols() const {return m_dims[1];}
    int GetNumElements() const { return m_numElements;}
    // TODO:
    int GetDepth() const {return 1;}
    int GetBatch() const {return 1;}
    bool IsEmpty() const {return (m_num_dims == 0);}

    // Return a copy of our buffer (Safe)
    vector<T> GetBufferCopy() const {return m_buffer;}

    // Iterators to manipulate the vector class member
    typename std::vector<T>::iterator begin();
    typename std::vector<T>::iterator end();
    typename std::vector<T>::const_iterator begin() const;
    typename std::vector<T>::const_iterator end() const;

    // Select a sub tensor from tensor
    template <typename ...Args>
    Tensor<T> Select(const range<int> &firstRange, Args... otherRange) const{
        int inputRows = this->GetRows();
        int inputCols = this->GetCols();                       
        int inputDepth = this->GetDepth();
        int inputBatch = this->GetBatch();

        Tensor<T> result;
        // Create array with all ranges (otherRange...) expand during compilation time
        array<range<int>, sizeof...(otherRange) + 1> pRanges = {firstRange, otherRange...};
        bool allRows = pRanges[0].empty();
        bool allCols = pRanges[1].empty();

        // Check if ranges don't go out of the matrix
        if ((pRanges[0].size() > inputRows) || (pRanges[1].size() > inputCols) || (pRanges[0].Min() > inputRows) || (pRanges[0].Max() > inputRows-1) || (pRanges[1].Min() > inputCols) || (pRanges[1].Max() > inputCols-1)){
            throw invalid_argument("Invalid select range");
        }

        if ((!allRows) && (allCols)){
            // Some rows, all cols
            result.SetDims(vector<int>{(int)pRanges[0].size(), inputCols});
            result.PreAloc();
            auto selRows = pRanges[0];
            auto startOffset = MAT_2D(selRows.Min(), 0, inputCols);
            auto start = m_buffer.begin()+startOffset;

            // Copy elements
            copy(start, start+result.GetNumElements(), result.begin());
        } else if ((allRows) && (!allCols)){
            // Some cols, all rows
            result.SetDims(vector<int>{inputRows, (int)pRanges[1].size()});
            result.PreAloc();
            auto selCols = pRanges[1];
            auto startOffset = MAT_2D(0, selCols.Min(), inputCols);
            // Get start iterator for result tensor
            auto startResult = result.begin();

            // Iterate on each line of m_buffer
            for (auto idx = m_buffer.begin()+startOffset; idx < m_buffer.end(); idx+=inputCols){
                // Copy x cols
                copy(idx, idx+selCols.size(), startResult);
                // Jump to next line
                startResult += selCols.size();
            }
        } else if ((!allRows) && (!allCols)){
            result.SetDims(vector<int>{(int)pRanges[0].size(), (int)pRanges[1].size()});
            result.PreAloc();
            auto selRows = pRanges[0];
            auto selCols = pRanges[1];
            auto startOffset = MAT_2D(selRows.Min(), selCols.Min(), inputCols);
            // Get start iterator for result tensor
            auto startResult = result.begin();

            /*auto start = m_buffer.begin()+startOffset;
            // Copy elements
            copy(start, start+result.GetNumElements(), result.begin());*/
            for (auto idx = m_buffer.begin()+startOffset; idx < m_buffer.end(); idx+=inputCols){
                copy(idx, idx+selCols.size(), startResult);
                startResult += selCols.size();                
                // Check if there is still space left to add
                if (startResult >= result.end()) break;
            }
        }
        return result;
    }

    /*
        Overload the "()" and "*" operators to make it feel like matlab
        The const after the method definition means that this method will not change the class members
    */

    // Address elements of tensor using a variadic template (Don't know how to add this on source)
    template <typename ...Args>
    T& operator()(int firstIdx, Args... otherIdx){
        // Create array with all parameters (otherIdx...) expand during compilation time
        array<int, sizeof...(otherIdx) + 1> pDims = {(int)firstIdx, (int)otherIdx...};
        switch (pDims.size()){
            case 1:
                return m_buffer.at(firstIdx);
            case 2:
                return m_buffer.at(MAT_2D(pDims[0], pDims[1], this->GetCols()));
            case 3:
                return m_buffer.at(MAT_3D(pDims[0], pDims[1], pDims[2], this->GetRows(), this->GetCols()));
            case 4:
                return m_buffer.at(MAT_4D(pDims[0], pDims[1], pDims[2], pDims[3], this->GetRows(), this->GetCols(), this->GetDepth()));
            default:
                throw invalid_argument("Addressing working up to 4 dimensions");
        }

        return m_buffer.at(firstIdx);
    }
    template <typename ...Args>
    T operator()(int firstIdx, Args... otherIdx) const{
        // Create array with all parameters (otherIdx...) expand during compilation time
        array<int, sizeof...(otherIdx) + 1> pDims = {(int)firstIdx, (int)otherIdx...};
        switch (pDims.size()){
            case 1:
                return m_buffer.at(firstIdx);
            case 2:
                return m_buffer.at(MAT_2D(pDims[0], pDims[1], this->GetCols()));
            case 3:
                return m_buffer.at(MAT_3D(pDims[0], pDims[1], pDims[2], this->GetRows(), this->GetCols()));
            case 4:
                return m_buffer.at(MAT_4D(pDims[0], pDims[1], pDims[2], pDims[3], this->GetRows(), this->GetCols(), this->GetDepth()));
            default:
                throw invalid_argument("Addressing working up to 4 dimensions");
        }

        return m_buffer.at(firstIdx);
    }

    Tensor<T> operator*(const Tensor &b) const;
    Tensor<T> operator*(const T b) const;
    Tensor<T> operator/(const T b) const;
    Tensor<T> operator+(const Tensor &b) const;
    Tensor<T> operator+(const T b) const;
    Tensor<T> operator-(const Tensor &b) const;
    Tensor<T> operator-(const T b) const;
    Tensor<T> operator-() const;
    Tensor<T> &operator=(const Tensor &b);
    bool operator==(const Tensor &b);
    // Implement the binary operators like Matlab
    Tensor<T> operator>=(const T &scalar);
    Tensor<T> operator<=(const T &scalar);
    Tensor<T> operator==(const T &scalar);
    Tensor<T> operator!=(const T &scalar);

    // Element-wise operations
    Tensor<T> EltWiseMult(const Tensor<T> &b) const;
    Tensor<T> EltWiseDiv(const Tensor<T> &b) const;

    /*
        Transpose (2d matrix only) and vanilla (with cache misses)
        For better implementation on CPU check here:
        http://stackoverflow.com/questions/9227747/in-place-transposition-of-a-matrix
        https://en.wikipedia.org/wiki/In-place_matrix_transposition#Algorithms
    */
    Tensor<T> Transpose() const;

    /*
        Repeat matrix on rows or cols
    */
    Tensor<T> Repmat(int nRows, int nCols) const;

    // A friend operator can see the private elements of this class
    friend Tensor<T> operator+(const T &left, const Tensor<T> &right){
        // Create result tensor with same dimensions
        Tensor<T> result(right.GetDims());

        // For each element of m_buffer multiply by b and store the result on resVec
        transform(right.begin(), right.end(), result.begin(),std::bind1st(plus<T>(),left));

        return result;
    }
    friend Tensor<T> operator-(const T &left, const Tensor<T> &right){
        // Create result tensor with same dimensions
        Tensor<T> result(right.GetDims());

        // For each element of m_buffer multiply by b and store the result on resVec
        transform(right.begin(), right.end(), result.begin(),std::bind1st(minus<T>(),left));

        return result;
    }
    friend Tensor<T> operator/(const T &left, const Tensor<T> &right){
        // Create result tensor with same dimensions
        Tensor<T> result(right.GetDims());

        // For each element of m_buffer multiply by b and store the result on resVec
        transform(right.begin(), right.end(), result.begin(),std::bind1st(std::divides<T>(),left));

        return result;
    }
    // Overload the << Operator to use this class with cout
    // https://msdn.microsoft.com/en-us/library/1z2f6c2k.aspx
    friend ostream& operator<<(ostream& os, const Tensor<T>& right){
        stringstream str_stream;
        str_stream << endl;
        if (right.IsEmpty()){
            // Empty tensor
            str_stream << "Empty Tensor:[]";
        } else {
            // Observe that the friend class can "see" elements from the Tensor class
            auto start = right.m_buffer.begin();
            auto ncols = right.GetCols();
            auto nrows = right.GetRows();
            for (int i = 0; i < nrows; ++i){
                // Get a slice from the vector
                vector<T> rowSlice(start, start + ncols);
                str_stream << "| ";
                for_each(rowSlice.begin(), rowSlice.end(), [&str_stream](T m){str_stream << m << " ";});
                start += ncols;
                str_stream << "|" << endl;
            }
        }
        //os << dt.mo << '/' << dt.da << '/' << dt.yr;
        os << str_stream.str();
        return os;
    }
};

#endif // TENSOR_H
