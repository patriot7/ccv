#include "ccv.h"
#include <sys/time.h>

//#define DEBUG

#define X_SLICE 4
#define Y_SLICE 3

unsigned int get_current_time(void);
void cos_ccv_slice_output(ccv_dense_matrix_t* mat, int x, int y);
ccv_dense_matrix_t* cos_ccv_merge(ccv_dense_matrix_t* mat[], int rows, int cols, int x, int y);

int
main(int argc, char** argv)
{
	ccv_enable_default_cache();

        unsigned int elapsed_time;
        ccv_dense_matrix_t* yuv = 0;
        ccv_read(argv[1], &yuv, CCV_IO_GRAY | CCV_IO_ANY_FILE);

        /* original edge detection time */
        ccv_dense_matrix_t* canny = 0;
        elapsed_time = get_current_time();
        ccv_canny(yuv, &canny, 0, 3, 175, 320);
        elapsed_time = get_current_time() - elapsed_time;
        printf("origin total: in time %d ms\n", elapsed_time);
        ccv_matrix_free(canny);

        ccv_dense_matrix_t* canny_arr[X_SLICE * Y_SLICE];
        int i, j, k = 0;

        /* sliced edge detection time */
        int slice_rows = yuv->rows / Y_SLICE;
        int slice_cols = yuv->cols / X_SLICE;
        elapsed_time = get_current_time();

#pragma omp parallel for private(j) shared(canny_arr, yuv)
        for (i = 0; i < Y_SLICE; i++) {
                ccv_dense_matrix_t* sliced_full_width = 0;
                ccv_slice(yuv, (ccv_matrix_t**)&sliced_full_width, 0, slice_rows * i, 0, slice_rows , yuv->cols);
                for (j = 0; j < X_SLICE; j++) {
                        ccv_dense_matrix_t* sliced_final = 0;
                        ccv_slice(sliced_full_width, (ccv_matrix_t**)&sliced_final, 0, 0, slice_cols * j, slice_rows, slice_cols);

#ifdef DEBUG
                        cos_ccv_slice_output(sliced_final, i, j);
#endif
                        canny_arr[k] = 0;
                        ccv_canny(sliced_final, &canny_arr[k], 0, 3, 175, 320);
                        k++;
                }
        }
        canny = 0;
        //canny = cos_ccv_merge(canny_arr, yuv->rows, yuv->cols, X_SLICE, Y_SLICE);
        elapsed_time = get_current_time() - elapsed_time;
        printf("sliced total: in time %d ms\n", elapsed_time);

        //ccv_matrix_free(canny);
        ccv_matrix_free(yuv);
	ccv_disable_cache();

	return 0;
}


ccv_dense_matrix_t*
cos_ccv_merge(ccv_dense_matrix_t* mat[], int rows, int cols, int x, int y)
{
        ccv_dense_matrix_t* merged = 0;
        merged = ccv_dense_matrix_new(rows, cols, mat[0]->type, 0, 0);

        int i, count = x * y, mrow, mcol, x_offset, y_offset;
        for (i = 0; i < count; i++) {
                x_offset = i % x;
                y_offset = i / x;
                for (mrow = 0; mrow < mat[i]->rows; mrow++) {
                        for (mcol = 0; mcol < mat[i]->step; mcol++) {
                                merged->data.u8[(y_offset * mat[i]->rows + mrow) * cols + x_offset * mat[i]->step + mcol] = mat[i]->data.u8[mrow * mat[i]->step + mcol];
                        }
                }
        }
#ifdef DEBUG
        int a, b;
        for (a = 0; a < merged->rows; a++) {
                for (b = 0; b < merged->step; b++) {
                        if (merged->data.u8[a * cols + b] != 0) merged->data.u8[a * cols + b] = 255;
                }
        }
        cos_ccv_slice_output(merged, 99, 99);
#endif
        return merged;
}

unsigned int
get_current_time(void)
{
	struct timeval tv;
	gettimeofday(&tv, 0);
	return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

void
cos_ccv_slice_output(ccv_dense_matrix_t* mat, int x, int y)
{
        char filename[11]; /* 99_99.jpeg max */
        sprintf(filename, "%d_%d.jpeg", x, y);
        ccv_write(mat, filename, 0, CCV_IO_JPEG_FILE, 0);
        return;
}
