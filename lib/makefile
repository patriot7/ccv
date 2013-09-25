include config.mk

#CC += -faddress-sanitizer -fno-omit-frame-pointer
CFLAGS := -O3 -ffast-math -Wall $(CFLAGS)# -fprofile-arcs -ftest-coverage

.PHONY: all clean

all: libccv.a

clean:
	rm -f *.o 3rdparty/sha1/*.o 3rdparty/sfmt/*.o 3rdparty/kissfft/*.o 3rdparty/dsfmt/*.o libccv.a

libccv.a: ccv_cache.o ccv_memory.o 3rdparty/sha1/sha1.o ccv_io.o  ccv_util.o ccv_basic.o ccv_resample.o  ccv_classic.o  ccv_bbf.o
	$(AR) rcs $@ $^

ccv_io.o: ccv_io.c ccv.h ccv_internal.h io/*.c
	$(CC) $< -o $@ -c $(CFLAGS)

%.o: %.c ccv.h ccv_internal.h
	$(CC) $< -o $@ -c $(CFLAGS)
