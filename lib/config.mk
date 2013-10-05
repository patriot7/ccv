CC := gcc
AR := ar
CFLAGS := -D HAVE_LIBJPEG -D HAVE_LIBPNG -D USE_OPENMP -fopenmp 
LDFLAGS := -lm -ljpeg -lpng -lz -lgomp 
