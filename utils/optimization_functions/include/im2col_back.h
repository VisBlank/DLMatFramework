/* Just declare function prototypes
Reference:
https://isocpp.org/wiki/faq/templates#separate-template-fn-defn-from-decl
*/

template <typename T>
void im2col_back(T *dout, int dout_H, int dout_W, int stride, int HH, int WW, int CC, T *img_grad, double *execTime, double *trfTime);
