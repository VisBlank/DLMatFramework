/*
* mex_matMult2D.c - Matlab bridge for im2col_back.
*
* Calculate im2col_back on an input image A.
*
* We are ROW MAJOR in C/C++ so you need to transpose the
* Output to get the result you would normally expect.
*
* The calling syntax is:
*
*		C = im2col_back(A, kernel_size, pad, stride)
*
*
* This is a MEX file for MATLAB.
*/

/* Mex headers */
#include "mex.h"

/* Define contract (Function prototype for matrix multiplication) */
#include "im2col_back.h"

/* The gateway function */

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

  if(nlhs != 3) {
    mexErrMsgIdAndTxt("matMul2D:nlhs",
    "Three outputs required.");
  }

  if(nrhs != 7) {
    mexErrMsgIdAndTxt("matMul2D:nrhs",
    "Five inputs required.");
  }

  bool isFloat = false;
  if (mxIsComplex(prhs[0])) {
    mexErrMsgIdAndTxt("im2col:isComplex","Input matrix must not be complex.");
  }
  // Detect the typename
  if ( !mxIsSingle(prhs[0]))  {
    isFloat = false;
  } else {
    isFloat = true;
  }


  /* variable declarations here */
  double *executionTime;
  double *transferTime;
  int inDout_H;
  int inDout_W;
  int height_out;
  int width_out;
  int chan_out;
  int nChan;
  int inStride;
  int nRowKer;
  int nColKer;

  /* read input data */
  inDout_H = mxGetScalar(prhs[1]);
  inDout_W = mxGetScalar(prhs[2]);
  inStride = mxGetScalar(prhs[3]);
  nRowKer = mxGetScalar(prhs[4]);
  nColKer = mxGetScalar(prhs[5]);
  nChan = mxGetScalar(prhs[6]);

  /* calculate output matrix dimensions */
  height_out = (inDout_H - 1) * inStride + nRowKer;
  width_out = (inDout_W - 1) * inStride + nColKer;
  chan_out = nChan;
  
  // Create execution time double scalar
  plhs[1] = mxCreateNumericMatrix(1,1,mxDOUBLE_CLASS,mxREAL);
  executionTime = (double *)mxGetPr(plhs[1]);

  // Create transfer time double scalar
  plhs[2] = mxCreateNumericMatrix(1,1,mxDOUBLE_CLASS,mxREAL);
  transferTime = (double *)mxGetPr(plhs[2]);

  // Define shape of output
  mwSize shape_out_mex[] = {height_out,width_out,chan_out};
  
  if (isFloat){
    /* creat ouput matrix as float(single)*/
    plhs[0] = mxCreateNumericArray(3,shape_out_mex,mxSINGLE_CLASS,mxREAL); //number 3 is temporary for now, should probably be replaced by a variable of dimensions!!
    float *outMatrix = (float *)mxGetPr(plhs[0]);
    im2col_back<float>((float *)mxGetPr(prhs[0]), inDout_H, inDout_W, inStride, nRowKer, nColKer, chan_out , outMatrix, executionTime, transferTime);
  } else {
    /* creat ouput matrix as float(double)*/
    plhs[0] = mxCreateNumericArray(3,shape_out_mex,mxDOUBLE_CLASS,mxREAL); //number 3 is temporary for now, should probably be replaced by a variable of dimensions!!
    double *outMatrix = (double *)mxGetPr(plhs[0]);
    im2col_back<double>((double *)mxGetPr(prhs[0]), inDout_H, inDout_W, inStride, nRowKer, nColKer, chan_out , outMatrix, executionTime, transferTime);
  }

}
