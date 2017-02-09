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

void im2col(float *data_im, int channels, int height, int width, int ky, int kx, int stride, int pad, float *data_col, double *execTime, double *trfTime) {

  /*Clear scren to help debug*/
  /*system("clear");*/

  /*
  We need a stride (positive and non-zero)
  */
  if (!stride) return;

  /* these tell us how many kernel windows we have across the input image */
  int height_col = (height + (2 * pad) - ky) / stride + 1;
  int width_col = (width + (2 * pad) - kx) / stride + 1;

  /* Detect fractional size convolution */
  float frac_height = ((float)height + (2.0 * (float)pad) - (float)ky) / (float)stride + 1.0;
  float frac_width = ((float)width + (2.0 * (float)pad) - (float)kx) / (float)stride + 1.0;
  int isFract = 0;
  if ( ((frac_height - (int)frac_height) == 0) && ((frac_width - (int)frac_width) == 0) ){
    isFract = 0;
  } else {
    isFract = 1;
  }

  /*printf("***************************NEW RUN******************************\n");
  printf("height=%d width=%d Stride=%d Pad=%d K=%d isFract=%d\n",height,width,stride,pad,kx,isFract);
  printf("height_col=%d width_col=%d\n",height_col,width_col);*/

  /*
  Calculate biggest row/col size considering padding or fractional convolution
  */
  int maxHeight = height - isFract + ((2 * pad)-1);
  int maxWidth = width - isFract + ((2 * pad)-1);
  /*printf("Maximum possible coordinates rows=%d Max cols=%d\n",maxHeight,maxWidth);*/

  /* Calculate how tall the collumn will be */
  int cols_height = channels * kx * ky;
  int kernelProd = kx * ky;

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

  /*
  * data_col
  * Will be a 2d matrix [cols_height x (height_col*width_col)]
  */
  int with_data_col = height_col * width_col;

  /* Measure elapsed wall time */
  struct timespec now, tmstart;
  /*tic*/
  clock_gettime(CLOCK_REALTIME, &tmstart);
  int channel,row,col;

  /* Iterate on the input image (could be virtually padded) */
  #pragma omp parallel for
  for (channel = 0; channel < channels; channel++) {
    /* Move down on the image */
    for (row = 0; row < height + (2 * pad); row += stride) {
      /* Move left on the image */
      for (col = 0; col < width + (2 * pad); col += stride) {
        /*printf("-----------------Coordinate [%d x %d]---------------\n",row,col);*/
        /*
        If the window is out of the image we should ignore the current
        iteration. But take care that this also may happen when we have
        padding, so check. Because if even with padding we go out of the
        window we should ignore(continue).
        */
        /*printf("Calculated window coverage: row=%d col=%d\n",(row + (ky-1) - 0),(col + (kx-1) - 0));*/
        if (((row + (ky-1)) > maxHeight) || ((col + (kx-1)) > maxWidth)) {
          /*printf("Out of window Coordinate [%d x %d] K:%d pad:%d Image[%d x %d]\n",row,col,kx,pad,height,width);*/
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
            Use on OpenCl
            int idxRow_out = (channel * (kernelProd)) + (m*ky) + n;
            */
            /*
            Fix offset if we're doing padding
            */
            int row_pad = (row + n) - pad;
            int col_pad = (col + m) - pad;
            /* Avoid running out of input image boundaries */
            if ((row_pad >= 0) && (col_pad >= 0) && (row_pad < height) && (col_pad < width)) {
              data_col[MAT_2D(idxRow_out, idxCol_out, cols_height)] =
              data_im[MAT_3D(row_pad, col_pad, channel, height, width)];
              /*printf("Adding valid elements %f\n",data_im[MAT_3D(row_pad, col_pad, channel, height, width)]);*/
            } else {
              /* If we're out return 0 */
              data_col[MAT_2D(idxRow_out, idxCol_out, cols_height)] = 0;

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

  /*toc*/
  clock_gettime(CLOCK_REALTIME, &now);
  double wall_time_sec =
  (double)((now.tv_sec + now.tv_nsec * 1e-9) -
  (double)(tmstart.tv_sec + tmstart.tv_nsec * 1e-9));
  *execTime = wall_time_sec;
  *trfTime = 0.0;
}
