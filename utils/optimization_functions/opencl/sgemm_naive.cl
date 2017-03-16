/*
  Define macros for row-major col-major 2d/3d matrix addressing.
*/
#ifdef USE_ROW_MAJOR
#define MAT_2D(i, j, width) (((i) * (width)) + (j))
#define MAT_3D(i, j, k, height, width)                                         \
(((i) * (width)) + (j) + ((k) * (height) * (width)))
#else
#define MAT_2D(i, j, height) (((j) * (height)) + (i))
#define MAT_3D(i, j, k, height, width)                                         \
(((j) * (height)) + (i) + ((k) * (height) * (width)))
#endif

/*
  __global const: Can help some OpenCl implementations to cache access
  restrict: helps to limit the effects of pointer aliasing,
  while often aiding the caching optimizations.
*/

__kernel void matrix_2d_mul_float_gpu(__global const float * restrict A, __global const float * restrict B,  __global const float * restrict biasVec, __global float * restrict C, int num_rows_A, int num_cols_A, int num_rows_B, int num_cols_B)
{
   int i = get_global_id(0);
   int k = get_global_id(1);


   // Sum is on the register(local to each thread)
  float sum = 0;

   // This iterate a lot on the global memory 2*j times
   //sum += A[i][j]*B[j][k];
  for (int j=0; j<num_cols_A; j++){
    #ifdef USE_ROW_MAJOR
      sum += A[MAT_2D(i,j,num_cols_A)] * B[MAT_2D(j,k,num_cols_B)];
      //printf("Coordinate [(i)%d x (k)%d] A[%d, %d]=%f B[%d, %d]=%f\n",i,k,i,j,A[MAT_2D(i,j,num_cols_A)],j,k,B[MAT_2D(j,k,num_cols_B)]);
    #else
      sum += A[MAT_2D(i,j,num_rows_A)] * B[MAT_2D(j,k,num_rows_B)];
      //printf("Coordinate [(i)%d x (k)%d] A[%d, %d]=%f B[%d, %d]=%f\n",i,k,i,j,A[MAT_2D(i,j,num_rows_A)],j,k,B[MAT_2D(j,k,num_rows_B)]);
    #endif
  }

  // Check how bias changes (for each i, or for each k)
  int bIdx = select(i, k, (num_rows_A == 1) );
  /*int bIdx = 0;
  if (num_rows_A == 1)
  {
    bIdx = k;
  }
  else
  {
    bIdx = i;
  }*/
  //printf("Coordinate [(i)%d x (k)%d] A[%d, %d] B[%d, %d] biasVec[%d]=%f\n",i,k,num_rows_A,num_cols_A,num_rows_B,num_cols_B,bIdx,biasVec[bIdx]);
  #ifdef USE_ROW_MAJOR
    C[MAT_2D(i,k,num_cols_B)]= sum + biasVec[bIdx];
  #else
    C[MAT_2D(i,k,num_rows_A)]= sum + biasVec[bIdx];
  #endif
}
