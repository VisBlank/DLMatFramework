#include "tensor.h"
#include "utils/hdf5tensor.h"
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
void Tensor<T>::SetDataFromBuffer(unique_ptr<T[]> ptrBuff){
    auto cont = 0;
    for (auto &it :m_buffer){
        it = ptrBuff[cont++];
    }
}

template<typename T>
void Tensor<T>::SaveToHDF5(const string &fileName) const{
    const string dataSpaceName = "Data";
    const string varName = "Tensor";
    HDF5Tensor<T>::WriteData(fileName, dataSpaceName, varName, (*this));
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
Tensor<T> Tensor<T>::EltWisePow(const T &scalar) const{
    // Create result tensor with same dimensions
    Tensor<T> result(vector<int>({m_dims}));

    // Do a pow for each element
    transform(m_buffer.begin(), m_buffer.end(), result.begin(),[&scalar](T m) -> T {return std::pow(m,scalar);});

    return result;
}

template<typename T>
Tensor<T> Tensor<T>::im2col(const Tensor<T> &input, int kx, int ky, int stride, int pad){
    // calculate output matrix dimensions
    int height_col = (input.GetRows() + (2 * pad) - ky) / stride + 1;
    int width_col = (input.GetCols() + (2 * pad) - kx) / stride + 1;
    auto cols_height = input.GetDepth()*kx*ky;
    auto cols_width = height_col * width_col;
    int kernelProd = kx * ky;

    // Detect fractional size convolution
    float frac_height = ((float)input.GetRows() + (2.0 * (float)pad) - (float)ky) / (float)stride + 1.0;
    float frac_width = ((float)input.GetCols() + (2.0 * (float)pad) - (float)kx) / (float)stride + 1.0;
    int isFract = 0;
    if ( ((frac_height - (int)frac_height) == 0) && ((frac_width - (int)frac_width) == 0) ){
        isFract = 0;
    } else {
        isFract = 1;
    }


    //Calculate biggest row/col size considering padding or fractional convolution
    int maxHeight = input.GetRows() - isFract + ((2 * pad)-1);
    int maxWidth = input.GetCols() - isFract + ((2 * pad)-1);

    /*
        Create variables to support richard formula, that calculates idxCol_out
        from (n_rows,n_cols,row,col,ky,kx,stride,width_col)
        Number of collumns that your slide window will cross on
        each dimension.
        Product k(x,y) * stride, calculating outside to avoid this multiplication
        for every row,col,channel
    */
    int n_cols = width_col * kx;
    int n_rows = height_col * ky;
    int prod_ky_stride = ky * stride;
    int prod_kx_stride = kx * stride;

    Tensor<T> result(vector<int>({cols_height,cols_width}));

    /* Iterate on the input image (could be virtually padded) */
    for (int channel = 0; channel < input.GetDepth(); channel++) {
        /* Move down on the image */
        for (int row = 0; row < input.GetRows() + (2 * pad); row += stride) {
            /* Move left on the image */
            for (int col = 0; col < input.GetCols() + (2 * pad); col += stride) {
                /*
                If the window is out of the image we should ignore the current
                iteration. But take care that this also may happen when we have
                padding, so check. Because if even with padding we go out of the
                window we should ignore(continue).
                */
                if (((row + (ky-1)) > maxHeight) || ((col + (kx-1)) > maxWidth)) {
                    continue;
                }

                /* Position the row of output channel related to the current channel */
                int idxRow_out = channel * (kernelProd);
                /*
                  X coordinate on output 2d matrix (Move right on output matrix)
                  Richard formula to calculate the collumn position of the output matrix
                  given (n_rows,n_cols,row,col,ky,kx,stride,width_col). Previously this
                  was calculated as "idxCol_out = (idxCol_out + 1) % with_data_col;"
                  after each window slide, but this was breaking full parallelization.
                */
                int idxCol_out = ((n_rows - (n_rows - row*ky))/(prod_ky_stride))*width_col + ((n_cols - (n_cols - col*kx)))/(prod_kx_stride) ;
                int m,n;
                /* Select window [ky x kx] on input volume on each channel */
                for (m = 0; m < ky; m++) {
                    for (n = 0; n < kx; n++) {
                        /*
                            Fix offset if we're doing padding
                        */
                        int row_pad = (row + n) - pad;
                        int col_pad = (col + m) - pad;
                        /* Avoid running out of input image boundaries */
                        if ((row_pad >= 0) && (col_pad >= 0) && (row_pad < input.GetRows()) && (col_pad < input.GetCols())) {
                            result(idxRow_out, idxCol_out) = input(row_pad, col_pad, channel);
                        } else {
                            /* If we're out return 0 */
                            result(idxRow_out, idxCol_out) = 0;

                        }
                        /*
                            Move down on the output 2d array to add current element
                            from the patch
                        */
                        idxRow_out++;
                    }
                }
            }
        }
    }

    return result;
}

template<typename T>
Tensor<T> Tensor<T>::im2col_back(const Tensor<T> &dout, int kx, int ky, int stride, int HH, int WW, int CC){
    int H = (dout.GetRows() - 1) * stride + ky;
    int W = (dout.GetCols() - 1) * stride + kx;

    Tensor<T> img_grad(vector<int>({H,W,CC}));

    auto dout_H = dout.GetRows();
    auto dout_W = dout.GetCols();

    //select patch
    #pragma omp parallel for
    for (int patchNum = 0; patchNum < (dout_H * dout_W); patchNum++){

      //starting upper left spatial coordinate for this patch
      int h_start = floor(((patchNum)/dout_W) * stride);
      int w_start = ( patchNum % dout_W ) * stride;

      //go over all the elements in selected patch placing/adding them into the output matrix
      int patchElement = 0; //counter for our current patch
      for (int channel = 0; channel < CC; channel++){

          for (int row = 0; row < HH; row++){

              for (int col = 0; col < WW; col++){

                  // Place patch on output, but increment values where patches overlap
                  // starting at col = (w_start * H) row = (h_start + row) go across the output, channel by channel (channel * H * W) and column by column (col * W)
                  //img_grad[(w_start * H) + (h_start + row) + (col * W) + (channel * H * W)] = img_grad[(w_start * H) + (h_start + row) + (col * W) + (channel * H * W)] + dout[patchNum + (patchElement * dout_H * dout_W)];
                  img_grad(h_start + row, w_start + col, channel) = img_grad(h_start + row, w_start + col, channel) + dout(patchNum + (patchElement * dout_H * dout_W));
                  patchElement++;
              }
          }
      }
    }
    return img_grad;
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
    // Iterate over the input matrix
    for (int r = 0; r < this->GetRows(); ++r){
        for (int c = 0; c < this->GetCols(); ++c){
            int resRows = r;
            // Number of times that this particular values should change
            for (int nrRep = 0; nrRep < nRows; ++nrRep){
                int resCols = c;
                for (int ncRep = 0; ncRep < nCols; ++ncRep){
                    result(resRows,resCols) = (*this)(r,c);
                    resCols += this->GetCols();
                }
                resRows += this->GetRows();
            }
        }
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
