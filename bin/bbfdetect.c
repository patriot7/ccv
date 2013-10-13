#include "ccv.h"
#include <sys/time.h>
#include <ctype.h>

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define DEBUG
//#define X_SLICE 5 
//#define Y_SLICE 7

unsigned int get_current_time(void);
void cos_ccv_slice_output(ccv_dense_matrix_t* mat, int x, int y);

int
main(int argc, char** argv)
{
        printf("Face Detection Benchmark ...\n");
	ccv_enable_default_cache();

	ccv_dense_matrix_t* image = 0;
	ccv_bbf_classifier_cascade_t* cascade = ccv_bbf_read_classifier_cascade(argv[4]);
	ccv_read(argv[1], &image, CCV_IO_GRAY | CCV_IO_ANY_FILE);

        unsigned int elapsed_time;
        ccv_array_t* seq;

        elapsed_time = get_current_time();
        seq = ccv_bbf_detect_objects(image, &cascade, 1, ccv_bbf_default_params);
        elapsed_time = get_current_time() - elapsed_time;
        printf("origin: %d in %dms\n", seq->rnum, elapsed_time);
        ccv_array_free(seq);

        int X_SLICE = atoi(argv[2]), Y_SLICE = atoi(argv[3]);
        int sliced_total = 0;
        int slice_rows = image->rows / Y_SLICE;
        int slice_cols = image->cols / X_SLICE;
        int i, count = X_SLICE * Y_SLICE;
        elapsed_time = get_current_time();
#pragma omp parallel for shared(sliced_total)
        for (i = 0; i < count; i++) {
                int y = i / X_SLICE;
                int x = i - X_SLICE * y;
                ccv_dense_matrix_t* slice = 0;
                ccv_slice(image, (ccv_matrix_t**)&slice, 0, slice_rows * y, slice_cols * x, slice_rows, slice_cols);
                ccv_array_t* sseq = ccv_bbf_detect_objects(slice, &cascade, 1, ccv_bbf_default_params);
                sliced_total += sseq->rnum;
#ifdef DEBUG
                cos_ccv_slice_output(slice, y, x);
#endif
        }
        elapsed_time = get_current_time() - elapsed_time;
        printf("slice & detect: %d in %dms\n", sliced_total, elapsed_time);

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
