#include <iostream>
#include <memory>
#include <tuple>

using namespace std;

template <typename T>
class Tensor {
protected:
  unique_ptr<T> m_Buffer;
public:
  // Constructor, the dimension is a variadic tuple
  template <typename Arg ...>
  Tensor(tuple<Arg...> dimns) {
    
  }

}
