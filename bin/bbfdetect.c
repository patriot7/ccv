#include "ccv.h"
#include <sys/time.h>
#include <ctype.h>

#include <stdio.h>
#include <omp.h>

#define FULL_HEIGHT 1993
#define FULL_WIDTH 3000
#define SLICE_V 6
#define SLICE_H 5
#define HEIGHT FULL_HEIGHT / SLICE_H
#define WIDTH FULL_WIDTH / SLICE_V

unsigned int get_current_time()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

int main(int argc, char** argv)
{
	assert(argc >= 3);
	int i, j;
	ccv_enable_default_cache();
	ccv_dense_matrix_t* image = 0;
	ccv_bbf_classifier_cascade_t* cascade = ccv_bbf_read_classifier_cascade(argv[2]);
	ccv_read(argv[1], &image, CCV_IO_GRAY | CCV_IO_ANY_FILE);

        unsigned int orig_elapsed_time, sliced_elapsed_time;
        ccv_array_t* seq;

        orig_elapsed_time = get_current_time();
        seq = ccv_bbf_detect_objects(image, &cascade, 1, ccv_bbf_default_params);
        orig_elapsed_time = get_current_time() - orig_elapsed_time;
        printf("origin total : %d in time %dms\n", seq->rnum, orig_elapsed_time);
        ccv_array_free(seq);

        int sliced_total = 0;
        sliced_elapsed_time = get_current_time();
#pragma omp parallel for private(j)
        for (i = 0; i < SLICE_H; i++) {
                ccv_dense_matrix_t* sliced_full_width = 0;
                ccv_slice(image, (ccv_matrix_t**)&sliced_full_width, 0, HEIGHT * i, 0, HEIGHT, FULL_WIDTH);
                for (j = 0; j < SLICE_V; j++) {
                        ccv_dense_matrix_t* sliced_final = 0;
                        ccv_slice(sliced_full_width, (ccv_matrix_t**)&sliced_final, 0, 0, WIDTH * j, HEIGHT, WIDTH);
                        //char filename[20];
                        //sprintf(filename, "%d_%d.jpeg", x, y);
                        //ccv_write(slice_area, filename, 0, CCV_IO_JPEG_FILE, 0);
                        seq = ccv_bbf_detect_objects(sliced_final, &cascade, 1, ccv_bbf_default_params);
                        //printf("pic %d_%d: %d face detected\n", x, y, seq_sliced->rnum);
                        sliced_total += seq->rnum;
                }
        }
        sliced_elapsed_time = get_current_time() - sliced_elapsed_time;
        printf("total : %d in time %dms\n", sliced_total, sliced_elapsed_time);
        ccv_array_free(seq);

        ccv_matrix_free(image);
	ccv_bbf_classifier_cascade_free(cascade);
	ccv_disable_cache();
	return 0;
}
