#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <cmath>
/*
Define macros to help addressing 1d arrays with 2d coordinates
both on row-major(C) or col-major(matlab)
With i:(row coordinate) and j:(col coordinate)
Row major (C)
A[i][j] == A[(i*num_cols_A)+j]
Col major (Matlab)
A[i][j] == A[(j*num_rows_A)+i]
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

template <typename T>
void im2col_back(T *dout, int dout_H, int dout_W, int stride, int HH, int WW, int CC, T *img_grad, double *execTime, double *trfTime) {

  /*Clear scren to help debug*/
  /*system("clear");*/

  
  /*calculate spatial dimensions of img_grad */
  
  int H = (dout_H - 1) * stride + HH;
  int W = (dout_W - 1) * stride + WW;
  
  int row,col,channel;
  
  /* Measure elapsed wall time */
  //struct timespec now, tmstart;
  /*tic*/
  //clock_gettime(CLOCK_REALTIME, &tmstart);

  
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
				  img_grad[(w_start * H) + (h_start + row) + (col * W) + (channel * H * W)] = img_grad[(w_start * H) + (h_start + row) + (col * W) + (channel * H * W)] + dout[patchNum + (patchElement * dout_H * dout_W)];  
                  patchElement++;
              } 
		  }
	  } 	  
  } 
  
 
    /*toc*/
  /*clock_gettime(CLOCK_REALTIME, &now);
  double wall_time_sec =
  (double)((now.tv_sec + now.tv_nsec * 1e-9) -
  (double)(tmstart.tv_sec + tmstart.tv_nsec * 1e-9));
  *execTime = wall_time_sec;
  *trfTime = 0.0;
*/
}

// Explicit declare available versions to avoid linker error.
template void im2col_back<double>(double *dout, int dout_H, int dout_W, int stride, int HH, int WW, int CC, double *img_grad, double *execTime, double *trfTime);
template void im2col_back<float>(float *dout, int dout_H, int dout_W, int stride, int HH, int WW, int CC, float *img_grad, double *execTime, double *trfTime);
