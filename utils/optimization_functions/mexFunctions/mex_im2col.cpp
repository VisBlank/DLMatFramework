/*
* mex_matMult2D.c - Matlab bridge for im2col.
*
* Calculate im2col on an input image A.
*
* We are ROW MAJOR in C/C++ so you need to transpose the
* Output to get the result you would normally expect.
*
* The calling syntax is:
*
*		C = im2col(A, kernel_size, pad, stride)
*
*
* This is a MEX file for MATLAB.
*/

/* Mex headers */
#include "mex.h"

/* Define contract (Function prototype for matrix multiplication) */
#include "im2col.h"

/* The gateway function */

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

  if(nlhs != 3) {
    mexErrMsgIdAndTxt("matMul2D:nlhs",
    "Three outputs required.");
  }

  if(nrhs != 5) {
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

  int * MatrixASize;
  int inKernelY;
  int inKernelX;
  int height_out;
  int width_out;
  int outMatrix_height;
  int outMatrix_width;
  int ncolsA;
  int nrowsA;
  int nchanA;
  int inPad;
  int inStride;
  int nDims;

  /* read input data */
  inKernelY = mxGetScalar(prhs[1]);
  inKernelX = mxGetScalar(prhs[2]);
  inStride = mxGetScalar(prhs[3]);
  inPad = mxGetScalar(prhs[4]);

  /* Get input matrix dimensions */
  MatrixASize = (int *)mxGetDimensions(prhs[0]);
  nDims = mxGetNumberOfDimensions (prhs[0]);

  if (nDims == 3)
  {
    ncolsA = MatrixASize[0];
    nrowsA = MatrixASize[1];
    nchanA = MatrixASize[2];
  }
  else if  (nDims == 2)
  {
    ncolsA = MatrixASize[0];
    nrowsA = MatrixASize[1];
    nchanA = 1;
  }

  /* calculate output matrix dimensions */
  height_out = (nrowsA + 2 * inPad - inKernelY) / inStride + 1;
  width_out = (ncolsA + 2 * inPad - inKernelX) / inStride + 1;

  outMatrix_height = nchanA*inKernelY*inKernelX;
  outMatrix_width = height_out * width_out;

  // Create execution time double scalar
  plhs[1] = mxCreateNumericMatrix(1,1,mxDOUBLE_CLASS,mxREAL);
  executionTime = (double *)mxGetPr(plhs[1]);

  // Create transfer time double scalar
  plhs[2] = mxCreateNumericMatrix(1,1,mxDOUBLE_CLASS,mxREAL);
  transferTime = (double *)mxGetPr(plhs[2]);

  if (isFloat){
    /* creat ouput matrix as float(single)*/
    plhs[0] = mxCreateNumericMatrix(outMatrix_height,outMatrix_width,mxSINGLE_CLASS,mxREAL);
    float *outMatrix = (float *)mxGetPr(plhs[0]);
    im2col<float>((float *)mxGetPr(prhs[0]), nchanA, nrowsA, ncolsA, inKernelY, inKernelX, inStride, inPad, outMatrix, executionTime, transferTime);
  } else {
    /* creat ouput matrix as float(single)*/
    plhs[0] = mxCreateNumericMatrix(outMatrix_height,outMatrix_width,mxDOUBLE_CLASS,mxREAL);
    double *outMatrix = (double *)mxGetPr(plhs[0]);
    im2col<double>((double *)mxGetPr(prhs[0]), nchanA, nrowsA, ncolsA, inKernelY, inKernelX, inStride, inPad, outMatrix, executionTime, transferTime);
  }

}
