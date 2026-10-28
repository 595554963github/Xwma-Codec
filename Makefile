CC=gcc
CFLAGS=-O2 -std=gnu11 -Wall

all: xwma

xwma: main.c xwma_encoder.c xwma_decoder.c wma_tables.h
	$(CC) $(CFLAGS) -o xwma main.c xwma_encoder.c xwma_decoder.c -lm

clean:
	rm -f xwma
