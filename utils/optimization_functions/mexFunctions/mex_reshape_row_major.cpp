/* Mex headers */
#include "mex.h"

/* Define contract (Function prototype for matrix reshape_row_major) */
#include "reshape_row_major.h"

/* The gateway function */
void mexFunction(int nlhs, mxArray *plhs[],
  int nrhs, const mxArray *prhs[])
  {
    /*verify input and output parameters*/
    if(nrhs != 2) {
      mexErrMsgIdAndTxt("mexReshapeRowMajor:nrhs",
      "Two inputs required.");
    }

    if(nlhs != 3) {
      mexErrMsgIdAndTxt("mexReshapeRowMajor:nlhs",
      "Three outputs required.");
    }

    /* make sure the second input argument is type single */
    if( !mxIsSingle(prhs[0]) || !mxIsClass(prhs[1],"double")
    || mxIsComplex(prhs[0]) || mxIsComplex(prhs[1])) {
      mexErrMsgIdAndTxt("mexReshapeRowMajor:notSingle","Input matrix must be type single, shape must be double");
    }

    /* check input matrix is 2D, maximal*/
    /*if( ((mxGetNumberOfDimensions(prhs[0])!=1) && (mxGetNumberOfDimensions(prhs[0])!=2)) ||
    ((mxGetNumberOfDimensions(prhs[1])!=1) && (mxGetNumberOfDimensions(prhs[1])!=2)) ) {
      mexErrMsgIdAndTxt("mexReshapeRowMajor:moreThan2D","Max dimension of input matrix is 2.");
    }*/

    /* variable declarations here */
    float * inMatrix;
    double *inShapeOut;

    float * outMatrix;
    double *executionTime;
    double *transferTime;

    mwSize ncolsA;
    mwSize nrowsA;
    mwSize nChannelsA = 1;
    mwSize nBatchesA = 1;
    mwSize ncolsShape;
    mwSize nrowsShape;

    /* read input data */
    inMatrix = (float *)mxGetPr(prhs[0]);
    inShapeOut = (double *)mxGetPr(prhs[1]);

    /* Get number of dimensions */
    mwSize nDimsA = mxGetNumberOfDimensions(prhs[0]);
    if (nDimsA > 2)
    {
      const mwSize *dims;
      dims = mxGetDimensions(prhs[0]);
      ncolsA = dims[0];
      nrowsA = dims[1];
      nChannelsA = dims[2];
      if (nDimsA > 3)
        nBatchesA = dims[3];
    }
    else
    {
      /* Get matrices dimensions */
      ncolsA = mxGetN(prhs[0]);
      nrowsA = mxGetM(prhs[0]);
    }

    ncolsShape = mxGetN(prhs[1]);
    nrowsShape = mxGetM(prhs[1]);

    int shape_out[] = {1,1,1,1};
    int shape_in[] = {nrowsA,ncolsA,nChannelsA,nBatchesA};
    mwSize shape_out_mex[] = {1,1,1,1};
    int cntShape = 0;
    for (cntShape = 0; cntShape < ncolsShape; cntShape++)
    {
        shape_out[cntShape] = (int)inShapeOut[cntShape];
        shape_out_mex[cntShape] = (mwSize)inShapeOut[cntShape];
    }

    /*
      Populate the shape array and get the output dimensions
    */
    int cnt = 0;
    int cnt_rev = 3;
    int outDim = 1;
    for (cnt = 0; cnt < nDimsA; cnt++)
    {
      shape_in[cnt] = (mxGetDimensions(prhs[0]))[cnt];
      if (shape_out[cnt_rev] != 1) outDim = cnt_rev + 1;
    }
    for (cnt = 3; cnt > 0; cnt--)
    {
      if (shape_out[cnt] != 1)
      {
        outDim = cnt + 1;
        break;
      }
    }

    /*printf("On MEX side---------------------\n");
    printf("Input Matrix: (dimensions=%d) Shape In [rows=%d cols=%d channels=%d batches=%d]\n",nDimsA, shape_in[0], shape_in[1], shape_in[2], shape_in[3]);
    printf("Desired shape: [%d %d %d %d]\n",shape_out[0], shape_out[1], shape_out[2], shape_out[3]);
    printf("Output dimension: %d\n",outDim);
    printf("--------------------------------\n");*/


    /* creat ouput matrix */
    /*plhs[0] = mxCreateNumericMatrix(nrowsA,ncolsA,mxSINGLE_CLASS,mxREAL);*/
    plhs[0] = mxCreateNumericArray(outDim,shape_out_mex,mxSINGLE_CLASS,mxREAL);
    outMatrix = (float *)mxGetPr(plhs[0]);

    plhs[1] = mxCreateNumericMatrix(1,1,mxDOUBLE_CLASS,mxREAL);
    executionTime = (double *)mxGetPr(plhs[1]);

    plhs[2] = mxCreateNumericMatrix(1,1,mxDOUBLE_CLASS,mxREAL);
    transferTime = (double *)mxGetPr(plhs[2]);


    /*Call some reshape_row_major_float implementation*/
    reshape_row_major_float(inMatrix, outMatrix, shape_out, shape_in ,executionTime, transferTime);


  }
