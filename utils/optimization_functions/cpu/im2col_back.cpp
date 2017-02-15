#include <stdio.h>
#include <time.h>
#include <stdlib.h>

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
  
  int H = (dout_H - 1) * S + HH;
  int W = (dout_W - 1) * S + WW;
  
  int row,col,channel
  
  /* Measure elapsed wall time */
  struct timespec now, tmstart;
  /*tic*/
  clock_gettime(CLOCK_REALTIME, &tmstart);

  
  //select patch
  for (int patchNum = 0; patchNum < (dout_H * dout_W); patchNum++){
	  //go over patch
	  for (int patchElement = 0; patchElement < HH * WW * CC; patchElement++){ 
		    
		  h_start = floor(((patchNum)/dout_W) * S);
		  w_start = ( patchNum % dout_W ) * S;
		  
		  //go over the output
		  for (int kernel_height_counter = h_start; kernel_height_counter < h_start + HH;kernel_height_counter++ ){
			  
			  for(int kernel_width_counter = w_start; kernel_width_counter < w_start + WW; kernel_width_counter++){
				img_grad[kernel_height_counter+kernel_width_counter] = img_grad[kernel_height_counter+kernel_width_counter] + dout[patchnum*(HH * WW * CC)+ patchElement];
			  }
		  
		  }
		  
	  }
	  
	  
	  
	  
  } 
  
    /*toc*/
  clock_gettime(CLOCK_REALTIME, &now);
  double wall_time_sec =
  (double)((now.tv_sec + now.tv_nsec * 1e-9) -
  (double)(tmstart.tv_sec + tmstart.tv_nsec * 1e-9));
  *execTime = wall_time_sec;
  *trfTime = 0.0;

}