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

using namespace std;

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
  unique_ptr<T[]> m_buffer;
  int m_num_dims;
  int m_numElements;
public:
  // Delete default Constructor
  Tensor() = delete;

  Tensor (const vector<int> &dims){
    m_num_dims = dims.size();
    // Do a "prod" of all elements on vector
    int prodDims = 1;
    for_each(dims.begin(), dims.end(), [&] (int m){prodDims *= m;});
    // Allocate memory for tensor buffer
    m_buffer = unique_ptr<T[]>(new T[prodDims]);
    m_numElements = prodDims;
  }

  /*void print() {

  }

  T& operator()(int row, int col){

  }

  const T& operator()(int row, int col) const{

  }*/

};


int main() {
  //testTupleIn(make_tuple(28,28,1,100));
  Tensor<float> testTensor(vector<int>({3,3}));

  testTensor.print();

  return 0;
}
