#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#include "reshape_row_major.h"

#ifdef USE_ROW_MAJOR
#define MAT_2D(i, j, width) (((i) * (width)) + (j))
#define MAT_3D(i, j, k, height, width) (((i) * (width)) + (j) + ((k) * (height) * (width)))
#define MAT_4D(i, j, k, l, height, width, depth) (((i) * (width)) + (j) + ((k) * (height) * (width)) + ((l) * (height) * (width) * (depth)))
#else
#define MAT_2D(i, j, height) (((j) * (height)) + (i))
#define MAT_3D(i, j, k, height, width) (((j) * (height)) + (i) + ((k) * (height) * (width)))
#define MAT_4D(i, j, k, l, height, width, depth) (((j) * (height)) + (i) + ((k) * (height) * (width)) + ((l) * (height) * (width) * (depth)))
#endif


void reshape_row_major_float(float * data_in, float * data_out, int shape_out[4], int shape_in[4], double *execTime, double *trfTime)
{
  int numRows_out = shape_out[0];
  int numCols_out = shape_out[1];
  int numChannels_out = shape_out[2];

  int numRows_in = shape_in[0];
  int numCols_in = shape_in[1];
  int numChannels_in = shape_in[2];
  int numBatches_in = shape_in[3];

  /* On Reshape the number of elements on the input is the same as the output */
  unsigned int numElementsIn = numRows_in*numCols_in*numChannels_in*numBatches_in;
  unsigned int N = 0;

  /* Measure elapsed wall time */
  struct timespec now, tmstart;
  /*tic*/
  clock_gettime(CLOCK_REALTIME, &tmstart);

  /* Iterate on the input signal */
  for (N = 0; N < numElementsIn; N++)
  {
    unsigned N_2d = N % (numRows_in*numCols_in);
    #ifdef USE_ROW_MAJOR
      unsigned int col_in = N_2d % numCols_in;
      unsigned int row_in = (N_2d - col_in) / numCols_in;
    #else
      unsigned int row_in = N_2d % numRows_in;
      unsigned int col_in = (N_2d - row_in) / numRows_in;
    #endif
    /* Calculate batch_in and channel_in from input N
      (Depends only on N and dims)
    */
    unsigned int channel_in = (N / (numCols_in*numRows_in)) % numChannels_in;
    unsigned int batch_in = (N / (numCols_in*numRows_in*numChannels_in)) % numBatches_in;

    /*
      Calculate output coordinates
    */

    /* Calculate output N (on row major order) */
    unsigned int N_row_major = ((row_in*numCols_in)+col_in) + (channel_in * numRows_in * numCols_in) + (batch_in * numRows_in * numCols_in * numChannels_in);

    /* Calculate channel and batch out From N_output */
    unsigned int channelOut = N_row_major / (numCols_out*numRows_out);
    unsigned int batchOut = N_row_major / (numCols_out*numRows_out*numChannels_out);

    /* Calculate output col/row */
    unsigned int colOut = ( col_in + (numCols_in*row_in) + (channel_in*numRows_in*numCols_in)) % numCols_out;
    unsigned int rowOut = (N_row_major - colOut - (channelOut*numCols_out*numRows_out) - (batchOut*numRows_out*numCols_out*numChannels_out)) / numCols_out;

    /* printf("N:%d -> data_in[row:%d, col:%d, channel:%d, batch:%d] = %f ***(reshape)*** data_out[row:%d, col:%d, channel:%d, batch:%d] --> N_out:(%d)\n",N, row_in, col_in, channel_in, batch_in, data_in[ N ], rowOut, colOut, channelOut, batchOut, N_row_major); */

    data_out[MAT_4D(rowOut,colOut,channelOut,batchOut,numRows_out,numCols_out,numChannels_out)] = data_in[ N ];
  }

  /*toc*/
  clock_gettime(CLOCK_REALTIME, &now);
  double wall_time_sec =
  (double)((now.tv_sec + now.tv_nsec * 1e-9) -
  (double)(tmstart.tv_sec + tmstart.tv_nsec * 1e-9));
  *execTime = wall_time_sec;
  *trfTime = 0.0;
}

void dispMatrix_float(float *matrix, int size[4])
{
  int rows,cols,channels,batches;
  int numRows = size[0];
  int numCols = size[1];
  int numChannels = size[2];
  int numBatches = size[3];

  printf("Matrix size [Rows:%d, Cols:%d, Channels:%d, Batches:%d]\n",numRows, numCols, numChannels, numBatches);

  for (batches = 0; batches < numBatches; batches++)
  {
    printf("Batch: %d\n\n",batches);
    for(channels=0;channels<numChannels;channels++)
    {
      printf("\t\t Channel: %d\n\n",channels);
      for(rows=0;rows<numRows;rows++)
      {
        printf("Row: %d -->  ",rows);
        for(cols=0;cols<numCols;cols++)
        {
          printf("| %3f  ",matrix[MAT_4D(rows,cols,channels,batches,numRows,numCols,numChannels)]);
        }
        printf("|\n");
      }
      printf("\n");
    }
    printf("------------------------------------\n");
  }
}
