#include "ccv.h"
#include <sys/time.h>
#include <ctype.h>

#include <stdio.h>
#include <omp.h>

//#define DEBUG

#define X_SLICE 7 
#define Y_SLICE 5 

unsigned int get_current_time(void);
void cos_ccv_slice_output(ccv_dense_matrix_t* mat, int x, int y);

int
main(int argc, char** argv)
{
	assert(argc >= 3);
	ccv_enable_default_cache();
	ccv_dense_matrix_t* image = 0;
	ccv_bbf_classifier_cascade_t* cascade = ccv_bbf_read_classifier_cascade(argv[2]);
	ccv_read(argv[1], &image, CCV_IO_GRAY | CCV_IO_ANY_FILE);

        unsigned int orig_elapsed_time, sliced_elapsed_time;
        ccv_array_t* seq;

        orig_elapsed_time = get_current_time();
        seq = ccv_bbf_detect_objects(image, &cascade, 1, ccv_bbf_default_params);
        orig_elapsed_time = get_current_time() - orig_elapsed_time;
        //printf("origin total : %d in time %dms\n", seq->rnum, orig_elapsed_time);
        printf("origin time: %dms\n", orig_elapsed_time);
        ccv_array_free(seq);

        int sliced_total = 0;
        int slice_rows = image->rows / Y_SLICE;
        int slice_cols = image->cols / X_SLICE;

        int x, y, i, count = X_SLICE * Y_SLICE;
        sliced_elapsed_time = get_current_time();
#pragma omp parallel for private(x, y)
        for (i = 0; i < count; i++) {
                y = i / X_SLICE;
                x = i - X_SLICE * y;
                ccv_dense_matrix_t* slice = 0;
                ccv_slice(image, (ccv_matrix_t**)&slice, 0, slice_rows * y, slice_cols * x, slice_rows, slice_cols);
                seq = ccv_bbf_detect_objects(slice, &cascade, 1, ccv_bbf_default_params);
                //sliced_total += seq->rnum;
#ifdef DEBUG
                cos_ccv_slice_output(sliced_final, x, y);
#endif
        }
        sliced_elapsed_time = get_current_time() - sliced_elapsed_time;
        //printf("total : %d in time %dms\n", sliced_total, sliced_elapsed_time);
        printf("openmp time: %dms\n", sliced_elapsed_time);
        ccv_array_free(seq);

        ccv_matrix_free(image);
	ccv_bbf_classifier_cascade_free(cascade);
	ccv_disable_cache();
	return 0;
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
