#include "ccv.h"
#include <sys/time.h>

//#define DEBUG

ccv_dense_matrix_t* cos_ccv_merge(ccv_dense_matrix_t* mat[], int rows, int cols, int x, int y);
int cos_ccv_count_edge_pixel(ccv_dense_matrix_t* mat);

unsigned int get_current_time()
{
	struct timeval tv;
	gettimeofday(&tv, 0);
	return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

int main(int argc, char** argv)
{
	ccv_enable_default_cache();

        unsigned int elapsed_time;
        int edge_count;
        ccv_dense_matrix_t* yuv = 0;
        ccv_read(argv[1], &yuv, CCV_IO_GRAY | CCV_IO_ANY_FILE);

        /* original edge detection time */
        ccv_dense_matrix_t* canny = 0;
        elapsed_time = get_current_time();
        ccv_canny(yuv, &canny, 0, 3, 175, 320);
        
        elapsed_time = get_current_time() - elapsed_time;
        printf("origin total: in time %d ms\n", elapsed_time);
        edge_count = cos_ccv_count_edge_pixel(canny);
        printf("original detected edge: %d\n", edge_count);
        /* clean variables for later use */
        ccv_matrix_free(canny);
        edge_count = 0;

        ccv_dense_matrix_t* canny_arr[4];
        int i, j, k = 0;

        /* sliced edge detection time */
        elapsed_time = get_current_time();
#ifndef DEBUG
#pragma omp parallel for private(j)
#endif
        for (i = 0; i < 2; i++) {
                ccv_dense_matrix_t* sliced_full_width = 0;
                ccv_slice(yuv, (ccv_matrix_t**)&sliced_full_width, 0, yuv->rows / 2 * i, 0, yuv->rows / 2 , yuv->cols);
                for (j = 0; j < 2; j++) {
                        ccv_dense_matrix_t* sliced_final = 0;
                        ccv_slice(sliced_full_width, (ccv_matrix_t**)&sliced_final, 0, 0, yuv->cols / 2 * j, yuv->rows / 2, yuv->cols / 2);
#ifdef DEBUG
                        char filename[20];
                        sprintf(filename, "%d_%d.jpeg", i, j);
                        ccv_write(sliced_final, filename, 0, CCV_IO_JPEG_FILE, 0);
#endif
                        canny_arr[k] = 0;
                        ccv_canny(sliced_final, &canny_arr[k], 0, 3, 175, 320);
#ifdef DEBUG
                        edge_count = cos_ccv_count_edge_pixel(canny_arr[k]);
                        printf("slice %d detected edge: %d\n", k, edge_count);
#endif
                        k++;
                }
        }
        canny = 0;
        canny = cos_ccv_merge(canny_arr, 500, 500, 2, 2);
        elapsed_time = get_current_time() - elapsed_time; /* should the time include the merging operation? */
        printf("sliced total: in time %d ms\n", elapsed_time);
        edge_count = cos_ccv_count_edge_pixel(canny);
        printf("sliced detected edge: %d\n", edge_count);

        ccv_matrix_free(canny);
        ccv_matrix_free(yuv);
	ccv_disable_cache();

	return 0;
}


ccv_dense_matrix_t*
cos_ccv_merge(ccv_dense_matrix_t* mat[], int rows, int cols, int x, int y)
{
        ccv_dense_matrix_t* merged = 0;

        merged = ccv_dense_matrix_new(rows, cols, CCV_8U | CCV_C1, 0, 0); /* type should get from original mats
                                                                             as we only use this function for
                                                                             edge detection, we just use 8U | C1 */

        int i, mrow, mcol, x_offset, y_offset, count = x * y;
        for (i = 0; i < count; i++) {
                x_offset = i % x;
                y_offset = i / y;
                for (mrow = 0; mrow < mat[i]->rows; mrow++) {
                        for (mcol = 0; mcol < mat[i]->cols; mcol++) {
                                merged->data.u8[(y_offset * mat[i]->rows + mrow) * cols + x_offset * mat[i]->cols + mcol] = mat[i]->data.u8[mrow * mat[i]->cols + mcol];
                        }
                }
        }

        return merged;
}

int
cos_ccv_count_edge_pixel(ccv_dense_matrix_t* mat)
{
        int i, counter = 0;
        for (i = 0; i < mat->rows * mat->cols; i++) {
                if (mat->data.u8[i] != 0) counter++;
        }
        return counter;
}
