#include "tensor.h"
/*
Tensor Implementation

Some references
http://supercomputingblog.com/openmp/tutorial-parallel-for-loops-with-openmp/
http://stackoverflow.com/questions/8384234/return-reference-to-a-vector-member-variable
https://mbevin.wordpress.com/2012/11/20/move-semantics/
http://stackoverflow.com/questions/115703/storing-c-template-function-definitions-in-a-cpp-file
*/

// Explicit declare available versions to avoid linker error.
//template class Tensor<float>;

template<typename T>
Tensor<T>::Tensor(const vector<int> &dims, initializer_list<T> list){
    m_buffer = vector<T>(list.size(),0);
    copy(list.begin(), list.end(), m_buffer.begin());
    SetDims(dims);
}

template<typename T>
Tensor<T>::Tensor(initializer_list<T> list){
    m_buffer = vector<T>(list.size(),0);
    copy(list.begin(), list.end(), m_buffer.begin());
}

template<typename T>
Tensor<T>::Tensor(const vector<int> &dims):m_dims(dims){
    m_num_dims = dims.size();
    SetDims(dims);
    // Initialize vector with prodDims size and fill with zeros
    m_buffer = vector<T>(m_numElements,0);
}

template<typename T>
void Tensor<T>::SetDims(const vector<int> &dims){
    m_dims = dims;
    m_num_dims = dims.size();
    // Do a "prod" of all elements on vector
    int prodDims = 1;
    for_each(dims.begin(), dims.end(), [&] (int m){prodDims *= m;});
    m_numElements = prodDims;
}

template<typename T>
void Tensor<T>::Reshape(const vector<int> &newdims){
    // Do a "prod" of all elements on vector
    int prodDims = 1;
    for_each(newdims.begin(), newdims.end(), [&] (int m){prodDims *= m;});
    if (prodDims != m_numElements){
        throw invalid_argument("Number of elements must be the same on new shape.");
    }
    m_dims = newdims;
}

template<typename T>
typename vector<T>::iterator Tensor<T>::begin(){
    return m_buffer.begin();
}

template<typename T>
typename vector<T>::iterator Tensor<T>::end() {
    return m_buffer.end();
}

template<typename T>
typename vector<T>::const_iterator Tensor<T>::begin() const {
    return m_buffer.begin();
}

template<typename T>
typename vector<T>::const_iterator Tensor<T>::end() const {
    return m_buffer.end();
}


template<typename T>
Tensor<T> Tensor<T>::operator+(const T b) const {
    // Create result tensor with same dimensions
    Tensor<T> result(vector<int>({m_dims}));

    // For each element of m_buffer multiply by b and store the result on resVec
    transform(m_buffer.begin(), m_buffer.end(), result.begin(),std::bind1st(std::plus<T>(),b));

    return result;
}

template<typename T>
Tensor<T> Tensor<T>::operator*(const T b) const{
    // Create result tensor with same dimensions
    Tensor<T> result(vector<int>({m_dims}));

    // For each element of m_buffer multiply by b and store the result on resVec
    transform(m_buffer.begin(), m_buffer.end(), result.begin(),std::bind1st(std::multiplies<T>(),b));

    return result;
}

template<typename T>
Tensor<T> Tensor<T>::operator-(const T b) const{
    // Create result tensor with same dimensions
    Tensor<T> result(vector<int>({m_dims}));

    // For each element of m_buffer multiply by b and store the result on resVec
    //transform(m_buffer.begin(), m_buffer.end(), result.begin(),std::bind1st(std::minus<T>(),b));
    transform(m_buffer.begin(), m_buffer.end(), result.begin(),[&b](T m)->T{ return m-b;});

    return result;
}

template<typename T>
Tensor<T> Tensor<T>::operator/(const T b) const{
    // Create result tensor with same dimensions
    Tensor<T> result(vector<int>({m_dims}));

    // For each element of m_buffer multiply by b and store the result on resVec
    //transform(m_buffer.begin(), m_buffer.end(), result.begin(),std::bind1st(std::divides<T>(),b));
    transform(m_buffer.begin(), m_buffer.end(), result.begin(),[&b](T m)->T{ return m/b;});

    return result;
}

template<typename T>
Tensor<T> Tensor<T>::operator*(const Tensor &b) const{
    vector<int> bDims = b.GetDims();
    if (b.GetNumDims() > 2){
        throw invalid_argument("Only 2d matrix multiplication is supported.");
    }
    // Also check if the multiplication is possible
    if (this->GetCols() != b.GetRows()){
        throw invalid_argument("A rows does not match to b cols.");
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

template<typename T>
Tensor<T> Tensor<T>::operator+(const Tensor &b) const {
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

template<typename T>
Tensor<T> Tensor<T>::operator-(const Tensor &b) const{
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

template<typename T>
Tensor<T> Tensor<T>::operator-() const{
    // Create result tensor with same dimensions
    Tensor<T> result = *this;

    // Negate all elements of result tensor
    transform (result.begin(), result.end(), result.begin(), negate<T>());

    return result;
}

template<typename T>
Tensor<T> &Tensor<T>::operator=(const Tensor &b){
    // Create result tensor with same dimensions
    //Tensor<T> result(vector<int>({m_dims}));
    //Tensor<T> result(b.GetDims());
    //copy(b.begin(), b.end(), result.begin());
    this->SetDims(b.GetDims());
    if (this->m_buffer.size() != this->m_numElements){
        m_buffer = vector<T>(m_numElements,0);
    }
    copy(b.begin(), b.end(), m_buffer.begin());
    return *this;
}

template<typename T>
bool Tensor<T>::operator==(const Tensor &b){

    bool result = true;

    //Check dims match
    if (b.GetDims() != this->GetDims()){
        throw invalid_argument("Dimensions must match.");
    }

    // Version with != operator (Overloaded on vector)
    /*if (this->GetBufferCopy() != b.m_buffer){
        result = false;
        return result;
    }*/
    // Version with std::equal
    if (!equal(this->begin(), this->end(), b.begin())){
        result = false;
        return result;
    }

    return result;
}

template<typename T>
Tensor<T> Tensor<T>::operator>=(const T &scalar){
    Tensor<T> result(this->GetDims());
    // Return the biggest element between scalar and in(each element)
    transform(m_buffer.begin(), m_buffer.end(), result.begin(),[&scalar](T m) -> T {return (m>=scalar)?(T)1:(T)0;});
    return result;
}

template<typename T>
Tensor<T> Tensor<T>::operator<=(const T &scalar){
    Tensor<T> result(this->GetDims());
    // Return the biggest element between scalar and in(each element)
    transform(m_buffer.begin(), m_buffer.end(), result.begin(),[&scalar](T m) -> T {return (m<=scalar)?(T)1:(T)0;});
    return result;
}

template<typename T>
Tensor<T> Tensor<T>::operator==(const T &scalar){
    Tensor<T> result(this->GetDims());
    // Return the biggest element between scalar and in(each element)
    transform(m_buffer.begin(), m_buffer.end(), result.begin(),[&scalar](T m) -> T {return (m==scalar)?(T)1:(T)0;});
    return result;
}

template<typename T>
Tensor<T> Tensor<T>::operator!=(const T &scalar){
    Tensor<T> result(this->GetDims());
    // Return the biggest element between scalar and in(each element)
    transform(m_buffer.begin(), m_buffer.end(), result.begin(),[&scalar](T m) -> T {return (m!=scalar)?(T)1:(T)0;});
    return result;
}

template<typename T>
Tensor<T> Tensor<T>::EltWiseMult(const Tensor<T> &b) const{
    if (GetDims() != b.GetDims()){
        throw invalid_argument("Both matrices must have same dimension.");
    }
    // Create result tensor with same dimensions
    Tensor<T> result(vector<int>({m_dims}));

    // Add contents of A and B and store results on resVec
    transform(m_buffer.begin(), m_buffer.end(), b.begin(),result.begin(), multiplies<T>());
    return result;
}

template<typename T>
Tensor<T> Tensor<T>::EltWiseDiv(const Tensor<T> &b) const{
    if (GetDims() != b.GetDims()){
        throw invalid_argument("Both matrices must have same dimension.");
    }
    // Create result tensor with same dimensions
    Tensor<T> result(vector<int>({m_dims}));

    // Add contents of A and B and store results on resVec
    transform(m_buffer.begin(), m_buffer.end(), b.begin(),result.begin(), divides<T>());

    return result;
}

template<typename T>
Tensor<T> Tensor<T>::Transpose() const{
    if (this->GetNumDims() > 2){
        throw invalid_argument("Only 2d matrix transpose is supported, use Permute for more dimensions");
    }
    // Create result tensor with reversed dimensions
    Tensor<T> result(vector<int>({m_dims.at(1), m_dims.at(0)}));

    // Lot's of cache-miss here
    #pragma omp parallel for
    for(int i=0; i<this->GetRows(); ++i) {
        for(int j=0; j<this->GetCols(); ++j) {
            result(j,i) = (*this)(i,j);
        }
    }
    return result;
}

template<typename T>
Tensor<T> Tensor<T>::Repmat(int nRows, int nCols) const{
    if (this->GetNumDims() > 2){
        throw invalid_argument("2d matrix support for repmat(TODO more dims)");
    }
    auto numReps = nRows*nCols;

    // Create result tensor
    Tensor<T> result(vector<int>({m_dims.at(0)*nRows, m_dims.at(1)*nCols}));

    // Repeat the content of "this" inside result numRep times
    auto offset = 0;
    for (auto i = 0; i < numReps; ++i){
        copy(m_buffer.begin(),m_buffer.end(),result.begin()+offset);
        // Increment with distance
        offset+=distance(m_buffer.begin(), m_buffer.end());
    }

    return result;
}

/*
 * Explicit declare template versions to avoid linker error. (This is needed if we use templates on .cpp files)
*/
template class Tensor<float>;
template class Tensor<double>;
template class Tensor<int>;
//template Tensor<float> Tensor<float>::crazy(int firstIdx, Args... otherIdx);
