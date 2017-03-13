/*
References
http://pt.cppreference.com/w/cpp/container/array
http://www.learncpp.com/cpp-tutorial/6-15-an-introduction-to-stdarray/
https://bitbucket.org/wlandry/ftensor
http://www.learncpp.com/cpp-tutorial/99-overloading-the-parenthesis-operator/
http://en.cppreference.com/w/cpp/algorithm/for_each
http://stackoverflow.com/questions/3221812/how-to-sum-up-elements-of-a-c-vector
https://www.programiz.com/cpp-programming/recursion
http://www.learncpp.com/cpp-tutorial/99-overloading-the-parenthesis-operator/
*/
#include <iostream>
#include <memory>
#include <tuple>

using namespace std;

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

  void print() {

  }

};
