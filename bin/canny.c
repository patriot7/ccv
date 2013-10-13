#include "ccv.h"
#include <sys/time.h>

//#define DEBUG

#define X_SLICE 7 
#define Y_SLICE 5 

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


        /* sliced edge detection time */
        int i, count = X_SLICE * Y_SLICE;
        int slice_rows = yuv->rows / Y_SLICE;
        int slice_cols = yuv->cols / X_SLICE;
        ccv_dense_matrix_t* canny_arr[count];
        elapsed_time = get_current_time();
#pragma omp parallel for
        for (i = 0; i < count; i++) {
                int y = i / X_SLICE;
                int x = i - X_SLICE * y;
                ccv_dense_matrix_t* slice = 0;
                ccv_slice(yuv, (ccv_matrix_t**)&slice, 0, slice_rows * y, slice_cols * x, slice_rows, slice_cols);
#ifdef DEBUG
                cos_ccv_slice_output(slice, y, x);
#endif
                canny_arr[i] = 0;
                ccv_canny(slice, &canny_arr[i], 0, 3, 175, 320);
        }
        elapsed_time = get_current_time() - elapsed_time;
        printf("slice: in time %d ms\n", elapsed_time);

        /* MERGE */
        elapsed_time = get_current_time();
        ccv_dense_matrix_t* final_output = 0;
        final_output = cos_ccv_merge(canny_arr, yuv->rows, yuv->cols, X_SLICE, Y_SLICE);
        elapsed_time = get_current_time() - elapsed_time;
        printf("merge: in time %d ms\n", elapsed_time);
        ccv_matrix_free(final_output);

        ccv_matrix_free(yuv);
	ccv_disable_cache();

	return 0;
}


ccv_dense_matrix_t*
cos_ccv_merge(ccv_dense_matrix_t* mat[], int rows, int cols, int x, int y)
{
        ccv_dense_matrix_t* merged = 0;
        merged = ccv_dense_matrix_new(rows, cols, mat[0]->type, 0, 0);

        unsigned long long i;
        int slice_rows = mat[0]->rows;
        int slice_step= mat[0]->step;
        int slice_num = x * y;
        unsigned long long pixel_num = slice_rows * slice_step; 
#pragma omp parallel for
        for (i = 0; i < slice_num; i++) {
                int x_offset = i % x;
                int y_offset = i / x;
                int j, mrow, mcol;
                for (j = 0; j < pixel_num; j++) {
                        mrow = j / slice_step;
                        mcol = j - slice_step * mrow;
                        merged->data.u8[(y_offset * slice_rows + mrow) * merged->step + x_offset * slice_step + mcol] = mat[i]->data.u8[mrow * slice_step + mcol];
                }
        }
#ifdef DEBUG
        unsigned long long merged_pixel_num = rows * cols;
        for (i = 0; i < merged_pixel_num; i++) {
                if (merged->data.u8[i] != 0) merged->data.u8[i] = 255;
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
