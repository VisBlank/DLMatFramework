/*
Main file

To compile:
g++ -std=c++11 main.cpp -o cppApp
*/
#include <iostream>
#include <tuple>
#include <vector>
#include <memory>
#include <algorithm>
#include <array>
#include <stdexcept>

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
  int m_num_dims;
  int m_numElements;
  vector<int> m_dims;
public:
  // Delete default Constructor
  Tensor() = delete;

  Tensor (const vector<int> &dims):m_dims(dims){
    m_num_dims = dims.size();
    // Do a "prod" of all elements on vector
    int prodDims = 1;
    for_each(dims.begin(), dims.end(), [&] (int m){prodDims *= m;});
    // Allocate memory for tensor buffer
    //m_buffer = unique_ptr<T[]>(new T[prodDims]);
    m_buffer.reserve(prodDims);
    m_numElements = prodDims;
  }

  // A const method does not change it's class members
  // The expected format will be rows,cols,channel,batch
  int GetNumDims() const{return m_num_dims;}
  vector<int> GetDims() const{return m_dims;}

  const vector<T> &GetVectorBuffer() const {return m_buffer;}

  void print() const{
    auto start = m_buffer.begin();
    for (int i = 0; i < m_dims[1]; ++i){
        // Get a slice from the vector
        vector<T> rowSlice(start, start + m_dims[1]);
        cout << "| ";
        for_each(rowSlice.begin(), rowSlice.end(), [](int m){cout << m << " ";});
        start += m_dims[1];
        cout << "|" << endl;
    }
  }

  /*
    Overload the "()" and "*" operators to make it feel like matlab
  */
  T& operator()(int row, int col){
    auto addr = MAT_2D(row, col,m_dims[1]);
    return m_buffer[addr];
  }

  const T& operator()(int row, int col) const{
    auto addr = MAT_2D(row, col,m_dims[1]);
    return m_buffer[addr];
  }

  Tensor<T> operator*(const Tensor &b){
      vector<int> bDims = b.GetDims();
      if (b.GetNumDims() > 2){
          throw invalid_argument("Only 2d matrix multiplication is supported.");
      }
      // Create result matrix
      Tensor<T> result(vector<int>({m_dims[0],bDims[1]}));
      int num_rows_A = m_dims[0];
      int num_cols_A = m_dims[1];
      int num_rows_B = bDims[0];
      int num_cols_B = bDims[1];
      const vector<T> &B = b.GetVectorBuffer();
      vector<T> C = result.GetVectorBuffer();
      for(int i=0; i<num_rows_A; ++i) {
          #pragma omp parallel for
          for (int k=0; k<num_cols_B; ++k) {
            T sum = 0;
            for (int j=0; j<num_cols_A; ++j){
                sum += m_buffer[MAT_2D(i,j,num_cols_A)] * B[MAT_2D(j,k,num_cols_B)];
            }
            //C[i*num_cols_B+k]=sum;
            result(i,k) = sum;
          }
      }
      return result;
  }

};


int main() {
  //testTupleIn(make_tuple(28,28,1,100));
  Tensor<float> A(vector<int>({2,2}));
  A(0,0) = 1;
  A(0,1) = 2;
  A(1,0) = 3;
  A(1,1) = 4;
  Tensor<float> B(vector<int>({2,2}));
  B(0,0) = 5;
  B(0,1) = 6;
  B(1,0) = 7;
  B(1,1) = 8;

  cout << "Matrix A:" << endl;
  A.print();
  cout << "Matrix B:" << endl;
  B.print();

  Tensor<float>C = A*B;
  cout << "A*B" << endl;
  C.print();

  return 0;
}
