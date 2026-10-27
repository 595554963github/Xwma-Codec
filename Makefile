CC=gcc
CFLAGS=-O2 -std=gnu11

all: xwma_encoder xwma_decoder

xwma_encoder: xwma_encoder.c wma_tables.h
	$(CC) $(CFLAGS) -o xwma_encoder xwma_encoder.c -lm

xwma_decoder: xwma_decoder.c wma_tables.h
	$(CC) $(CFLAGS) -o xwma_decoder xwma_decoder.c -lm

clean:
	rm -f xwma_encoder xwma_decoder