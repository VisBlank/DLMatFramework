/*
* mex_matMult2D.c - Matlab bridge for various matrix 2d implementation.
* This function will allow you to call directly from matlab some matrix
* 2d implementation
*
* Multiplies 2 matrices A[MxN], B[NxO], resulting on C[MxO]
*
* The calling syntax is:
*
*		C = mex_matMult2D(A, B)
*
* This is a MEX file for MATLAB.
*/

/* Mex headers */
#include "mex.h"

/* Define contract (Function prototype for matrix multiplication) */
#include "matrix_2d_mul.h"

/* The gateway function */
void mexFunction(int nlhs, mxArray *plhs[],
  int nrhs, const mxArray *prhs[])
  {
    /*verify input and output parameters*/
    if(nrhs != 3) {
      mexErrMsgIdAndTxt("matMul2D:nrhs",
      "Three inputs required.");
    }

    if(nlhs != 3) {
      mexErrMsgIdAndTxt("matMul2D:nlhs",
      "Three outputs required.");
    }

    /* make sure the second input argument is type single */
    if( !mxIsSingle(prhs[0]) || !mxIsSingle(prhs[1])
    || mxIsComplex(prhs[0]) || mxIsComplex(prhs[1])) {
      mexErrMsgIdAndTxt("matMul2D:notSingle","Input matrix must be type single.");
    }

    /* check input matrix is 2D, maximal*/
    if( ((mxGetNumberOfDimensions(prhs[0])!=1) && (mxGetNumberOfDimensions(prhs[0])!=2)) ||
    ((mxGetNumberOfDimensions(prhs[1])!=1) && (mxGetNumberOfDimensions(prhs[1])!=2)) ) {
      mexErrMsgIdAndTxt("matMul2D:moreThan2D","Max dimension of input matrix is 2.");
    }

    /* ncolsA should equal to nrowsB*/
    if (mxGetN(prhs[0])!=mxGetM(prhs[1])) {
      mexErrMsgIdAndTxt("matMul2D:mismatchDimension",
      "For A*B, col num of A should equal to row num of B.");
    }

    /* variable declarations here */
    float * inMatrixA;
    float * inMatrixB;
    float * inMatrix_biasVec;

    float * outMatrix;
    double *executionTime;
    double *transferTime;

    mwSize ncolsA;
    mwSize nrowsA;
    mwSize ncolsB;
    mwSize nrowsB;
    mwSize ncolsC;
    mwSize nrowsC;
    mwSize ncolsBias;
    mwSize nrowsBias;

    /* read input data */
    inMatrixA = (float *)mxGetPr(prhs[0]);
    inMatrixB = (float *)mxGetPr(prhs[1]);
    inMatrix_biasVec = (float *)mxGetPr(prhs[2]);

    /* Get matrices dimensions */
    ncolsA = mxGetN(prhs[0]);
    nrowsA = mxGetM(prhs[0]);
    ncolsB = mxGetN(prhs[1]);
    nrowsB = mxGetM(prhs[1]);
    ncolsBias = mxGetN(prhs[2]);
    nrowsBias = mxGetM(prhs[2]);

    /* creat ouput matrix */
    plhs[0] = mxCreateNumericMatrix(nrowsA,ncolsB,mxSINGLE_CLASS,mxREAL);
    outMatrix = (float *)mxGetPr(plhs[0]);

    plhs[1] = mxCreateNumericMatrix(1,1,mxDOUBLE_CLASS,mxREAL);
    executionTime = (double *)mxGetPr(plhs[1]);

    plhs[2] = mxCreateNumericMatrix(1,1,mxDOUBLE_CLASS,mxREAL);
    transferTime = (double *)mxGetPr(plhs[2]);


    /*
    * Call some implementation of matrix_2d_mul_float. The idea is that you
    * could swap implementations if they follow same contract. Here contract
    * just means same function name and parameters
    */
    matrix_2d_mul_float(inMatrixA, inMatrixB, inMatrix_biasVec, outMatrix,
      nrowsA, ncolsA,
      nrowsB, ncolsB,
      nrowsBias*ncolsBias, executionTime, transferTime);



    }
