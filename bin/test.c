#include "ccv.h"
#include <stdio.h>
#include <stdlib.h>

extern void ccv_add(ccv_matrix_t* a, ccv_matrix_t* b, ccv_matrix_t** c, int type);

int
main(int argc, char** argv)
{
        ccv_dense_matrix_t* mat1 = 0;
        ccv_dense_matrix_t* mat2 = 0;
        ccv_dense_matrix_t* mat3 = 0;

        ccv_read(argv[1], &mat1, CCV_IO_GRAY | CCV_IO_ANY_FILE);
        ccv_read(argv[2], &mat2, CCV_IO_GRAY | CCV_IO_ANY_FILE);

        ccv_add((ccv_matrix_t *)mat1, (ccv_matrix_t *)mat2, (ccv_matrix_t **)&mat3, mat1->type);
        ccv_write(mat3, "test.jpeg", 0, CCV_IO_JPEG_FILE, 0);

        ccv_matrix_free(mat1);
        ccv_matrix_free(mat2);
        ccv_matrix_free(mat3);

        return 0;
}
