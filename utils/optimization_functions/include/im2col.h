/* Just declare function prototypes
Reference:
https://isocpp.org/wiki/faq/templates#separate-template-fn-defn-from-decl
*/

template <typename T>
void im2col(T* data_im, int channels,  int height,  int width, int ky, int kx, int stride, int pad, T* data_col, double *execTime, double *trfTime);
