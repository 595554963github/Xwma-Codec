#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

void xwma_encode_file(const char *in_path, const char *out_path, int bit_rate);
int xwma_decode_xwma(const uint8_t *data, int size, const char *outpath);
int xwma_encoder_selftest(void);
int xwma_decoder_selftest(void);

static uint8_t *read_whole_file(const char *path, int *out_size)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        return NULL;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fprintf(stderr, "cannot seek %s\n", path);
        fclose(f);
        return NULL;
    }
    long sz = ftell(f);
    if (sz < 0) {
        fprintf(stderr, "cannot size %s\n", path);
        fclose(f);
        return NULL;
    }
    rewind(f);
    uint8_t *buf = (uint8_t *)malloc((size_t)sz);
    if (!buf) {
        fprintf(stderr, "out of memory\n");
        fclose(f);
        return NULL;
    }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        fprintf(stderr, "read error %s\n", path);
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *out_size = (int)sz;
    return buf;
}

static void default_output(const char *in, const char *ext, char *out, size_t out_cap)
{
    const char *dot = strrchr(in, '.');
    size_t n = dot ? (size_t)(dot - in) : strlen(in);
    size_t need = strlen(ext) + 1;
    if (n + need > out_cap)
        n = out_cap - need;
    memcpy(out, in, n);
    memcpy(out + n, ext, strlen(ext));
    out[n + strlen(ext)] = 0;
}

static void usage(const char *name)
{
    fprintf(stderr, "usage:\n");
    fprintf(stderr, "  %s -e <input.wav> [output.xwma] [bitrate]\n", name);
    fprintf(stderr, "  %s -d <input.xwma> [output.wav]\n", name);
    fprintf(stderr, "  %s -t\n", name);
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        usage(argc > 0 ? argv[0] : "xwma");
        return 1;
    }

    if (!strcmp(argv[1], "-t"))
        return xwma_encoder_selftest() | xwma_decoder_selftest();

    if (argc < 3 || argv[2][0] == '-') {
        usage(argv[0]);
        return 1;
    }

    if (!strcmp(argv[1], "-e")) {
        int bitrate = 96000;
        if (argc > 4) {
            bitrate = atoi(argv[4]);
            if (bitrate < 8000) {
                fprintf(stderr, "bitrate too low\n");
                return 1;
            }
        }
        char out[4096];
        if (argc > 3)
            snprintf(out, sizeof(out), "%s", argv[3]);
        else
            default_output(argv[2], ".xwma", out, sizeof(out));
        xwma_encode_file(argv[2], out, bitrate);
        return 0;
    }

    if (!strcmp(argv[1], "-d")) {
        char out[4096];
        if (argc > 3)
            snprintf(out, sizeof(out), "%s", argv[3]);
        else
            default_output(argv[2], ".wav", out, sizeof(out));
        int size = 0;
        uint8_t *data = read_whole_file(argv[2], &size);
        if (!data)
            return 1;
        int rc = xwma_decode_xwma(data, size, out);
        free(data);
        return rc < 0 ? 1 : 0;
    }

    usage(argv[0]);
    return 1;
}
