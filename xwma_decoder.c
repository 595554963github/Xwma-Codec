#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <sys/stat.h>
#include <dirent.h>

#define BLOCK_MIN_BITS 7
#define BLOCK_MAX_BITS 11
#define BLOCK_MAX_SIZE (1 << BLOCK_MAX_BITS)
#define BLOCK_NB_SIZES (BLOCK_MAX_BITS - BLOCK_MIN_BITS + 1)
#define HIGH_BAND_MAX_SIZE 16
#define NB_LSP_COEFS 10
#define MAX_CODED_SUPERFRAME_SIZE 32768
#define MAX_CHANNELS 2
#define NOISE_TAB_SIZE 8192
#define LSP_POW_BITS 7
#define VLCBITS 9
#define VLCMAX ((22 + VLCBITS - 1) / VLCBITS)
#define EXPVLCBITS 8
#define EXPMAX ((19 + EXPVLCBITS - 1) / EXPVLCBITS)
#define HGAINVLCBITS 9
#define HGAINMAX ((13 + HGAINVLCBITS - 1) / HGAINVLCBITS)
#define PAD 64

typedef struct {
    const uint8_t *buf;
    int size_bits;
    int index;
} BitReader;

typedef struct {
    int ch[2];
    int sym;
} VlcNode;

typedef struct {
    VlcNode *nodes;
    int count;
    int cap;
} VLC;

typedef struct {
    int n;
    int max_level;
    const uint32_t *huffcodes;
    const uint8_t *huffbits;
    const uint16_t *levels;
} CoefVLCTable;

typedef struct {
    int n;
    double *tw1r;
    double *tw1i;
    double *thr;
    double *thi;
    double *fr;
    double *fi;
} MDCT;

#include "wma_tables.h"

typedef struct {
    BitReader gb;
    int version;
    int use_bit_reservoir;
    int use_variable_block_len;
    int use_exp_vlc;
    int use_noise_coding;
    int byte_offset_bits;
    VLC exp_vlc;
    int exponent_sizes[BLOCK_NB_SIZES];
    uint16_t exponent_bands[BLOCK_NB_SIZES][25];
    int high_band_start[BLOCK_NB_SIZES];
    int coefs_start;
    int coefs_end[BLOCK_NB_SIZES];
    int exponent_high_sizes[BLOCK_NB_SIZES];
    int exponent_high_bands[BLOCK_NB_SIZES][HIGH_BAND_MAX_SIZE];
    VLC hgain_vlc;
    int high_band_coded[MAX_CHANNELS][HIGH_BAND_MAX_SIZE];
    int high_band_values[MAX_CHANNELS][HIGH_BAND_MAX_SIZE];
    VLC coef_vlc[2];
    uint16_t *run_table[2];
    float *level_table[2];
    uint16_t *int_table[2];
    const CoefVLCTable *coef_vlcs[2];
    int frame_len;
    int frame_len_bits;
    int nb_block_sizes;
    int reset_block_lengths;
    int block_len_bits;
    int next_block_len_bits;
    int prev_block_len_bits;
    int block_len;
    int block_num;
    int block_pos;
    int ms_stereo;
    uint8_t channel_coded[MAX_CHANNELS];
    int exponents_bsize[MAX_CHANNELS];
    float exponents[MAX_CHANNELS][BLOCK_MAX_SIZE];
    float max_exponent[MAX_CHANNELS];
    float coefs1[MAX_CHANNELS][BLOCK_MAX_SIZE];
    float coefs[MAX_CHANNELS][BLOCK_MAX_SIZE];
    double output[BLOCK_MAX_SIZE * 2];
    MDCT mdct[BLOCK_NB_SIZES];
    const float *windows[BLOCK_NB_SIZES];
    float *winbuf[BLOCK_NB_SIZES];
    float frame_out[MAX_CHANNELS][BLOCK_MAX_SIZE * 2];
    uint8_t last_superframe[MAX_CODED_SUPERFRAME_SIZE + PAD];
    int last_bitoffset;
    int last_superframe_len;
    int exponents_initialized[MAX_CHANNELS];
    float noise_table[NOISE_TAB_SIZE];
    int noise_index;
    float noise_mult;
    float lsp_cos_table[BLOCK_MAX_SIZE];
    float lsp_pow_e_table[256];
    float lsp_pow_m_table1[(1 << LSP_POW_BITS)];
    float lsp_pow_m_table2[(1 << LSP_POW_BITS)];
    int channels;
    int sample_rate;
    int bit_rate;
    int block_align;
    float *pcm[MAX_CHANNELS];
    int pcm_len;
    int pcm_cap;
} WMAContext;

static int av_log2(unsigned int v)
{
    return 31 - __builtin_clz(v);
}

static void br_init(BitReader *b, const uint8_t *buf, int bits)
{
    b->buf = buf;
    b->size_bits = bits;
    b->index = 0;
}

static inline uint32_t br_get(BitReader *b, int n)
{
    const uint8_t *p = b->buf + (b->index >> 3);
    int bit = b->index & 7;
    uint64_t w = ((uint64_t)p[0] << 56) | ((uint64_t)p[1] << 48) |
                 ((uint64_t)p[2] << 40) | ((uint64_t)p[3] << 32) |
                 ((uint64_t)p[4] << 24) | ((uint64_t)p[5] << 16) |
                 ((uint64_t)p[6] << 8) | (uint64_t)p[7];
    uint32_t v = (uint32_t)((w >> (64 - n - bit)) & (n >= 32 ? 0xFFFFFFFFu : ((1u << n) - 1)));
    b->index += n;
    return v;
}

static inline int br_get1(BitReader *b)
{
    return (int)br_get(b, 1);
}

static inline void br_skip(BitReader *b, int n)
{
    b->index += n;
}

static inline int br_count(BitReader *b)
{
    return b->index;
}

static inline int br_left(BitReader *b)
{
    return b->size_bits - b->index;
}

static inline void br_align(BitReader *b)
{
    b->index = (b->index + 7) & ~7;
}

static void vlc_alloc(VLC *v, int maxnodes)
{
    v->nodes = (VlcNode *)malloc(sizeof(VlcNode) * maxnodes);
    v->count = 0;
    v->cap = maxnodes;
    v->nodes[0].ch[0] = v->nodes[0].ch[1] = -1;
    v->nodes[0].sym = -1;
    v->count = 1;
}

static void vlc_free(VLC *v)
{
    free(v->nodes);
    v->nodes = NULL;
}

static void vlc_insert(VLC *v, uint32_t code, int len, int sym)
{
    int node = 0;
    for (int i = len - 1; i >= 0; i--) {
        int bit = (code >> i) & 1;
        if (v->nodes[node].ch[bit] < 0) {
            v->nodes[node].ch[bit] = v->count;
            v->nodes[v->count].ch[0] = v->nodes[v->count].ch[1] = -1;
            v->nodes[v->count].sym = -1;
            v->count++;
        }
        node = v->nodes[node].ch[bit];
    }
    v->nodes[node].sym = sym;
}

static void vlc_init_codes(VLC *v, const uint8_t *bits, const uint32_t *codes, int n)
{
    vlc_alloc(v, n * 24 + 2);
    for (int i = 0; i < n; i++) {
        if (bits[i] > 0)
            vlc_insert(v, codes[i], bits[i], i);
    }
}

static void vlc_init_canonical(VLC *v, const uint8_t *bits, const int *syms, int n)
{
    uint32_t code = 0;
    int prev = 0;
    vlc_alloc(v, n * 16 + 2);
    for (int i = 0; i < n; i++) {
        int len = bits[i];
        code <<= (len - prev);
        vlc_insert(v, code, len, syms[i]);
        code++;
        prev = len;
    }
}

static inline int vlc_get(BitReader *b, VLC *v)
{
    int node = 0;
    while (v->nodes[node].sym < 0) {
        int bit = br_get1(b);
        node = v->nodes[node].ch[bit];
        if (node < 0)
            return -1;
    }
    return v->nodes[node].sym;
}

static void fft(double *fr, double *fi, int n, int sign)
{
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j |= bit;
        if (i < j) {
            double t = fr[i]; fr[i] = fr[j]; fr[j] = t;
            t = fi[i]; fi[i] = fi[j]; fi[j] = t;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        double ang = sign * 2.0 * M_PI / len;
        double wr = cos(ang), wi = sin(ang);
        for (int i = 0; i < n; i += len) {
            double cr = 1.0, ci = 0.0;
            for (int k = 0; k < len / 2; k++) {
                int a = i + k, b2 = i + k + len / 2;
                double vr = fr[b2] * cr - fi[b2] * ci;
                double vi = fr[b2] * ci + fi[b2] * cr;
                fr[b2] = fr[a] - vr;
                fi[b2] = fi[a] - vi;
                fr[a] += vr;
                fi[a] += vi;
                double ncr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = ncr;
            }
        }
    }
}

static void mdct_init(MDCT *m, int n)
{
    m->n = n;
    int half = n / 2;
    m->tw1r = (double *)malloc(sizeof(double) * half * 2);
    m->tw1i = (double *)malloc(sizeof(double) * half * 2);
    m->thr = (double *)malloc(sizeof(double) * n * 2);
    m->thi = (double *)malloc(sizeof(double) * n * 2);
    m->fr = (double *)malloc(sizeof(double) * n * 2);
    m->fi = (double *)malloc(sizeof(double) * n * 2);
    for (int j = 0; j < half; j++) {
        double a = M_PI * j * (n + 1) / (2.0 * n);
        m->tw1r[j] = cos(a);
        m->tw1i[j] = sin(a);
    }
    for (int k = 0; k < n * 2; k++) {
        double a = M_PI * (k + 0.5) / (2.0 * n) + M_PI / 4.0;
        m->thr[k] = cos(a);
        m->thi[k] = sin(a);
    }
}

static void mdct_free(MDCT *m)
{
    free(m->tw1r);
    free(m->tw1i);
    free(m->thr);
    free(m->thi);
    free(m->fr);
    free(m->fi);
}

static void mdct_run(MDCT *m, double *out, const float *in)
{
    int n = m->n;
    int half = n / 2;
    int len = n * 2;
    double *fr = m->fr, *fi = m->fi;
    memset(fr, 0, sizeof(double) * len);
    memset(fi, 0, sizeof(double) * len);
    for (int j = 0; j < half; j++) {
        double a = in[j];
        double b = in[n - 1 - j];
        double tr = m->tw1r[j], ti = m->tw1i[j];
        fr[j] = a * tr;
        fi[j] = a * ti;
        fr[n + j] = b * ti;
        fi[n + j] = -b * tr;
    }
    fft(fr, fi, len, 1);
    {
        double *xv = m->fi;
        for (int k = 0; k < len; k++)
            xv[k] = fr[k] * m->thr[k] - fi[k] * m->thi[k];
        for (int i = 0; i < half; i++)
            out[i] = -xv[i] / 32768.0;
        for (int i = half; i < n; i++)
            out[i] = xv[n - 1 - i] / 32768.0;
        for (int i = n; i < n + half; i++)
            out[i] = -xv[i] / 32768.0;
        for (int i = n + half; i < len; i++)
            out[i] = -xv[len + n - 1 - i] / 32768.0;
    }
}

static void init_coef_vlc(VLC *vlc, uint16_t **prun_table, float **plevel_table,
                          uint16_t **pint_table, const CoefVLCTable *vlc_table)
{
    int n = vlc_table->n;
    const uint8_t *table_bits = vlc_table->huffbits;
    const uint32_t *table_codes = vlc_table->huffcodes;
    const uint16_t *levels_table = vlc_table->levels;
    uint16_t *run_table, *int_table;
    float *flevel_table;
    int i, l, j, k, level;

    vlc_init_codes(vlc, table_bits, table_codes, n);

    run_table = (uint16_t *)malloc(sizeof(uint16_t) * n);
    flevel_table = (float *)malloc(sizeof(float) * n);
    int_table = (uint16_t *)malloc(sizeof(uint16_t) * n);
    i = 2;
    level = 1;
    k = 0;
    while (i < n) {
        int_table[k] = i;
        l = levels_table[k++];
        for (j = 0; j < l; j++) {
            run_table[i] = j;
            flevel_table[i] = (float)level;
            i++;
        }
        level++;
    }
    *prun_table = run_table;
    *plevel_table = flevel_table;
    *pint_table = int_table;
}

static int wma_init(WMAContext *s, int channels, int sample_rate, int bit_rate,
                    int block_align, int flags2)
{
    int i;
    float bps1, high_freq;
    float bps;
    int sample_rate1;
    int coef_vlc_table;

    if (sample_rate <= 0 || sample_rate > 50000 ||
        channels <= 0 || channels > 2 ||
        bit_rate <= 0)
        return -1;

    s->channels = channels;
    s->sample_rate = sample_rate;
    s->bit_rate = bit_rate;
    s->block_align = block_align;
    s->version = 2;

    s->use_exp_vlc = flags2 & 0x0001;
    s->use_bit_reservoir = flags2 & 0x0002;
    s->use_variable_block_len = flags2 & 0x0004;

    for (i = 0; i < MAX_CHANNELS; i++)
        s->max_exponent[i] = 1.0;

    s->frame_len_bits = 11;
    if (sample_rate <= 16000)
        s->frame_len_bits = 9;
    else if (sample_rate <= 22050)
        s->frame_len_bits = 10;
    s->next_block_len_bits = s->frame_len_bits;
    s->prev_block_len_bits = s->frame_len_bits;
    s->block_len_bits = s->frame_len_bits;

    s->frame_len = 1 << s->frame_len_bits;
    if (s->use_variable_block_len) {
        int nb_max, nb;
        nb = ((flags2 >> 3) & 3) + 1;
        if ((bit_rate / channels) >= 32000)
            nb += 2;
        nb_max = s->frame_len_bits - BLOCK_MIN_BITS;
        if (nb > nb_max)
            nb = nb_max;
        s->nb_block_sizes = nb + 1;
    } else
        s->nb_block_sizes = 1;

    s->use_noise_coding = 1;
    high_freq = sample_rate * 0.5;

    sample_rate1 = sample_rate;
    if (sample_rate1 >= 44100)
        sample_rate1 = 44100;
    else if (sample_rate1 >= 22050)
        sample_rate1 = 22050;
    else if (sample_rate1 >= 16000)
        sample_rate1 = 16000;
    else if (sample_rate1 >= 11025)
        sample_rate1 = 11025;
    else if (sample_rate1 >= 8000)
        sample_rate1 = 8000;

    bps = (float)bit_rate / (float)(channels * sample_rate);
    s->byte_offset_bits = av_log2((int)(bps * s->frame_len / 8.0 + 0.5)) + 2;
    if (s->byte_offset_bits + 3 > 25)
        return -1;

    bps1 = bps;
    if (channels == 2)
        bps1 = bps * 1.6;
    if (sample_rate1 == 44100) {
        if (bps1 >= 0.61)
            s->use_noise_coding = 0;
        else
            high_freq = high_freq * 0.4;
    } else if (sample_rate1 == 22050) {
        if (bps1 >= 1.16)
            s->use_noise_coding = 0;
        else if (bps1 >= 0.72)
            high_freq = high_freq * 0.7;
        else
            high_freq = high_freq * 0.6;
    } else if (sample_rate1 == 16000) {
        if (bps > 0.5)
            high_freq = high_freq * 0.5;
        else
            high_freq = high_freq * 0.3;
    } else if (sample_rate1 == 11025)
        high_freq = high_freq * 0.7;
    else if (sample_rate1 == 8000) {
        if (bps <= 0.625)
            high_freq = high_freq * 0.5;
        else if (bps > 0.75)
            s->use_noise_coding = 0;
        else
            high_freq = high_freq * 0.65;
    } else {
        if (bps >= 0.8)
            high_freq = high_freq * 0.75;
        else if (bps >= 0.6)
            high_freq = high_freq * 0.6;
        else
            high_freq = high_freq * 0.5;
    }

    {
        int a, b, pos, lpos, k, block_len, j, n;
        const uint8_t *table;

        s->coefs_start = 0;
        for (k = 0; k < s->nb_block_sizes; k++) {
            block_len = s->frame_len >> k;

            table = NULL;
            a = s->frame_len_bits - BLOCK_MIN_BITS - k;
            if (a < 3) {
                if (sample_rate >= 44100)
                    table = exponent_band_44100[a];
                else if (sample_rate >= 32000)
                    table = exponent_band_32000[a];
                else if (sample_rate >= 22050)
                    table = exponent_band_22050[a];
            }
            if (table) {
                n = *table++;
                for (i = 0; i < n; i++)
                    s->exponent_bands[k][i] = table[i];
                s->exponent_sizes[k] = n;
            } else {
                j = 0;
                lpos = 0;
                for (i = 0; i < 25; i++) {
                    a = ff_wma_critical_freqs[i];
                    b = sample_rate;
                    pos = ((block_len * 2 * a) + (b << 1)) / (4 * b);
                    pos <<= 2;
                    if (pos > block_len)
                        pos = block_len;
                    if (pos > lpos)
                        s->exponent_bands[k][j++] = pos - lpos;
                    if (pos >= block_len)
                        break;
                    lpos = pos;
                }
                s->exponent_sizes[k] = j;
            }

            s->coefs_end[k] = (s->frame_len - ((s->frame_len * 9) / 100)) >> k;
            s->high_band_start[k] = (int)((block_len * 2 * high_freq) / sample_rate + 0.5);
            n = s->exponent_sizes[k];
            j = 0;
            pos = 0;
            for (i = 0; i < n; i++) {
                int start, end;
                start = pos;
                pos += s->exponent_bands[k][i];
                end = pos;
                if (start < s->high_band_start[k])
                    start = s->high_band_start[k];
                if (end > s->coefs_end[k])
                    end = s->coefs_end[k];
                if (end > start)
                    s->exponent_high_bands[k][j++] = end - start;
            }
            s->exponent_high_sizes[k] = j;
        }
    }

    for (i = 0; i < s->nb_block_sizes; i++) {
        int len = 1 << (s->frame_len_bits - i);
        float *w = (float *)malloc(sizeof(float) * len);
        for (int j2 = 0; j2 < len; j2++)
            w[j2] = (float)sin((j2 + 0.5) * M_PI / (2.0 * len));
        s->winbuf[i] = w;
        s->windows[i] = w;
    }

    s->reset_block_lengths = 1;

    if (s->use_noise_coding) {
        unsigned int seed;
        float norm;
        if (s->use_exp_vlc)
            s->noise_mult = 0.02;
        else
            s->noise_mult = 0.04;
        seed = 1;
        norm = (1.0 / (float)(1LL << 31)) * sqrt(3) * s->noise_mult;
        for (i = 0; i < NOISE_TAB_SIZE; i++) {
            seed = seed * 314159 + 1;
            s->noise_table[i] = (float)((int)seed) * norm;
        }
    }

    coef_vlc_table = 2;
    if (sample_rate >= 32000) {
        if (bps1 < 0.72)
            coef_vlc_table = 0;
        else if (bps1 < 1.16)
            coef_vlc_table = 1;
    }
    s->coef_vlcs[0] = &coef_vlcs[coef_vlc_table * 2];
    s->coef_vlcs[1] = &coef_vlcs[coef_vlc_table * 2 + 1];
    init_coef_vlc(&s->coef_vlc[0], &s->run_table[0], &s->level_table[0],
                  &s->int_table[0], s->coef_vlcs[0]);
    init_coef_vlc(&s->coef_vlc[1], &s->run_table[1], &s->level_table[1],
                  &s->int_table[1], s->coef_vlcs[1]);

    for (i = 0; i < s->nb_block_sizes; i++)
        mdct_init(&s->mdct[i], 1 << (s->frame_len_bits - i));

    if (s->use_noise_coding) {
        int syms[37];
        uint8_t lens[37];
        for (i = 0; i < 37; i++) {
            lens[i] = ff_wma_hgain_hufftab[i][1];
            syms[i] = ff_wma_hgain_hufftab[i][0] - 18;
        }
        vlc_init_canonical(&s->hgain_vlc, lens, syms, 37);
    }

    if (s->use_exp_vlc) {
        vlc_init_codes(&s->exp_vlc, ff_aac_scalefactor_bits, ff_aac_scalefactor_code, 121);
    } else {
        float wdel, a2, b2;
        wdel = M_PI / s->frame_len;
        for (i = 0; i < s->frame_len; i++)
            s->lsp_cos_table[i] = 2.0f * (float)cos(wdel * i);
        for (i = 0; i < 256; i++) {
            int e = i - 126;
            s->lsp_pow_e_table[i] = (float)exp2(e * -0.25);
        }
        b2 = 1.0f;
        for (i = (1 << LSP_POW_BITS) - 1; i >= 0; i--) {
            int m = (1 << LSP_POW_BITS) + i;
            a2 = (float)m * (0.5f / (1 << LSP_POW_BITS));
            a2 = 1 / sqrtf(sqrtf(a2));
            s->lsp_pow_m_table1[i] = 2 * a2 - b2;
            s->lsp_pow_m_table2[i] = b2 - a2;
            b2 = a2;
        }
    }

    return 0;
}

static void wma_end(WMAContext *s)
{
    int i;
    for (i = 0; i < s->nb_block_sizes; i++) {
        mdct_free(&s->mdct[i]);
        free(s->winbuf[i]);
    }
    if (s->use_exp_vlc)
        vlc_free(&s->exp_vlc);
    if (s->use_noise_coding)
        vlc_free(&s->hgain_vlc);
    for (i = 0; i < 2; i++) {
        vlc_free(&s->coef_vlc[i]);
        free(s->run_table[i]);
        free(s->level_table[i]);
        free(s->int_table[i]);
    }
}

static int wma_total_gain_to_bits(int total_gain)
{
    if (total_gain < 15)
        return 13;
    else if (total_gain < 32)
        return 12;
    else if (total_gain < 40)
        return 11;
    else if (total_gain < 45)
        return 10;
    else
        return 9;
}

static int wma_run_level_decode(WMAContext *s, BitReader *gb, VLC *vlc,
                                const float *level_table, const uint16_t *run_table,
                                float *ptr, int num_coefs, int block_len,
                                int frame_len_bits, int coef_nb_bits)
{
    int code, level, sign, offset;
    const uint32_t *ilvl = (const uint32_t *)level_table;
    const unsigned int coef_mask = block_len - 1;
    for (offset = 0; offset < num_coefs; offset++) {
        code = vlc_get(gb, vlc);
        if (code > 1) {
            offset += run_table[code];
            sign = br_get1(gb) - 1;
            {
                union { float f; uint32_t u; } t;
                t.u = ilvl[code] ^ (sign & 0x80000000);
                ptr[offset & coef_mask] = t.f;
            }
        } else if (code == 1) {
            break;
        } else {
            level = br_get(gb, coef_nb_bits);
            offset += br_get(gb, frame_len_bits);
            sign = br_get1(gb) - 1;
            ptr[offset & coef_mask] = (float)((level ^ sign) - sign);
        }
    }
    if (offset > num_coefs)
        return -1;
    return 0;
}

static inline float pow_m1_4(WMAContext *s, float x)
{
    union {
        float f;
        unsigned int v;
    } u, t;
    unsigned int e, m;
    float a, b;

    u.f = x;
    e = u.v >> 23;
    m = (u.v >> (23 - LSP_POW_BITS)) & ((1 << LSP_POW_BITS) - 1);
    t.v = ((u.v << LSP_POW_BITS) & ((1 << 23) - 1)) | (127 << 23);
    a = s->lsp_pow_m_table1[m];
    b = s->lsp_pow_m_table2[m];
    return s->lsp_pow_e_table[e] * (a + b * t.f);
}

static void wma_lsp_to_curve(WMAContext *s, float *out, float *val_max_ptr,
                             int n, float *lsp)
{
    int i, j;
    float p, q, w, v, val_max;

    val_max = 0;
    for (i = 0; i < n; i++) {
        p = 0.5f;
        q = 0.5f;
        w = s->lsp_cos_table[i];
        for (j = 1; j < NB_LSP_COEFS; j += 2) {
            q *= w - lsp[j - 1];
            p *= w - lsp[j];
        }
        p *= p * (2.0f - w);
        q *= q * (2.0f + w);
        v = p + q;
        v = pow_m1_4(s, v);
        if (v > val_max)
            val_max = v;
        out[i] = v;
    }
    *val_max_ptr = val_max;
}

static void decode_exp_lsp(WMAContext *s, int ch)
{
    float lsp_coefs[NB_LSP_COEFS];
    int val, i;

    for (i = 0; i < NB_LSP_COEFS; i++) {
        if (i == 0 || i >= 8)
            val = br_get(&s->gb, 3);
        else
            val = br_get(&s->gb, 4);
        lsp_coefs[i] = ff_wma_lsp_codebook[i][val];
    }

    wma_lsp_to_curve(s, s->exponents[ch], &s->max_exponent[ch],
                     s->block_len, lsp_coefs);
}

static int decode_exp_vlc(WMAContext *s, int ch)
{
    int last_exp, n, code;
    const uint16_t *ptr;
    float v, max_scale;
    float *q, *q_end;
    const float *ptab = pow_tab + 60;

    ptr = s->exponent_bands[s->frame_len_bits - s->block_len_bits];
    q = s->exponents[ch];
    q_end = q + s->block_len;
    max_scale = 0;
    last_exp = 36;

    while (q < q_end) {
        code = vlc_get(&s->gb, &s->exp_vlc);
        last_exp += code - 60;
        if ((unsigned)last_exp + 60 >= sizeof(pow_tab) / sizeof(pow_tab[0]))
            return -1;
        v = ptab[last_exp];
        if (v > max_scale)
            max_scale = v;
        n = *ptr++;
        while (n-- > 0)
            *q++ = v;
    }
    s->max_exponent[ch] = max_scale;
    return 0;
}

static void wma_window(WMAContext *s, float *out)
{
    double *in = s->output;
    int block_len, bsize, n, i;

    if (s->block_len_bits <= s->prev_block_len_bits) {
        block_len = s->block_len;
        bsize = s->frame_len_bits - s->block_len_bits;
        for (i = 0; i < block_len; i++)
            out[i] += (float)(in[i] * s->windows[bsize][i]);
    } else {
        block_len = 1 << s->prev_block_len_bits;
        n = (s->block_len - block_len) / 2;
        bsize = s->frame_len_bits - s->prev_block_len_bits;
        for (i = 0; i < block_len; i++)
            out[n + i] += (float)(in[n + i] * s->windows[bsize][i]);
        for (i = 0; i < n; i++)
            out[n + block_len + i] = (float)in[n + block_len + i];
    }

    out += s->block_len;
    in += s->block_len;

    if (s->block_len_bits <= s->next_block_len_bits) {
        block_len = s->block_len;
        bsize = s->frame_len_bits - s->block_len_bits;
        for (i = 0; i < block_len; i++)
            out[i] = (float)(in[i] * s->windows[bsize][block_len - 1 - i]);
    } else {
        block_len = 1 << s->next_block_len_bits;
        n = (s->block_len - block_len) / 2;
        bsize = s->frame_len_bits - s->next_block_len_bits;
        for (i = 0; i < n; i++)
            out[i] = (float)in[i];
        for (i = 0; i < block_len; i++)
            out[n + i] = (float)(in[n + i] * s->windows[bsize][block_len - 1 - i]);
        for (i = 0; i < n; i++)
            out[n + block_len + i] = 0;
    }
}

static int wma_decode_block(WMAContext *s)
{
    int channels = s->channels;
    int n, v, a, ch, bsize;
    int coef_nb_bits, total_gain;
    int nb_coefs[MAX_CHANNELS];
    float mdct_norm;

    if (s->use_variable_block_len) {
        n = av_log2(s->nb_block_sizes - 1) + 1;

        if (s->reset_block_lengths) {
            s->reset_block_lengths = 0;
            v = br_get(&s->gb, n);
            if (v >= s->nb_block_sizes)
                return -1;
            s->prev_block_len_bits = s->frame_len_bits - v;
            v = br_get(&s->gb, n);
            if (v >= s->nb_block_sizes)
                return -1;
            s->block_len_bits = s->frame_len_bits - v;
        } else {
            s->prev_block_len_bits = s->block_len_bits;
            s->block_len_bits = s->next_block_len_bits;
        }
        v = br_get(&s->gb, n);
        if (v >= s->nb_block_sizes)
            return -1;
        s->next_block_len_bits = s->frame_len_bits - v;
    } else {
        s->next_block_len_bits = s->frame_len_bits;
        s->prev_block_len_bits = s->frame_len_bits;
        s->block_len_bits = s->frame_len_bits;
    }

    if (s->frame_len_bits - s->block_len_bits >= s->nb_block_sizes)
        return -1;

    s->block_len = 1 << s->block_len_bits;
    if ((s->block_pos + s->block_len) > s->frame_len)
        return -1;

    if (channels == 2)
        s->ms_stereo = br_get1(&s->gb);
    v = 0;
    for (ch = 0; ch < channels; ch++) {
        a = br_get1(&s->gb);
        s->channel_coded[ch] = a;
        v |= a;
    }

    bsize = s->frame_len_bits - s->block_len_bits;

    if (!v)
        goto next;

    total_gain = 1;
    for (;;) {
        if (br_left(&s->gb) < 7)
            return -1;
        a = br_get(&s->gb, 7);
        total_gain += a;
        if (a != 127)
            break;
    }

    coef_nb_bits = wma_total_gain_to_bits(total_gain);

    n = s->coefs_end[bsize] - s->coefs_start;
    for (ch = 0; ch < channels; ch++)
        nb_coefs[ch] = n;

    if (s->use_noise_coding) {
        for (ch = 0; ch < channels; ch++) {
            if (s->channel_coded[ch]) {
                int i, nn;
                nn = s->exponent_high_sizes[bsize];
                for (i = 0; i < nn; i++) {
                    a = br_get1(&s->gb);
                    s->high_band_coded[ch][i] = a;
                    if (a)
                        nb_coefs[ch] -= s->exponent_high_bands[bsize][i];
                }
            }
        }
        for (ch = 0; ch < channels; ch++) {
            if (s->channel_coded[ch]) {
                int i, nn, val;

                nn = s->exponent_high_sizes[bsize];
                val = (int)0x80000000;
                for (i = 0; i < nn; i++) {
                    if (s->high_band_coded[ch][i]) {
                        if (val == (int)0x80000000) {
                            val = br_get(&s->gb, 7) - 19;
                        } else {
                            val += vlc_get(&s->gb, &s->hgain_vlc);
                        }
                        s->high_band_values[ch][i] = val;
                    }
                }
            }
        }
    }

    if ((s->block_len_bits == s->frame_len_bits) || br_get1(&s->gb)) {
        for (ch = 0; ch < channels; ch++) {
            if (s->channel_coded[ch]) {
                if (s->use_exp_vlc) {
                    if (decode_exp_vlc(s, ch) < 0)
                        return -1;
                } else {
                    decode_exp_lsp(s, ch);
                }
                s->exponents_bsize[ch] = bsize;
                s->exponents_initialized[ch] = 1;
            }
        }
    }

    for (ch = 0; ch < channels; ch++) {
        if (s->channel_coded[ch] && !s->exponents_initialized[ch])
            return -1;
    }

    for (ch = 0; ch < channels; ch++) {
        if (s->channel_coded[ch]) {
            int tindex;
            float *ptr = &s->coefs1[ch][0];
            int ret;

            tindex = (ch == 1 && s->ms_stereo);
            memset(ptr, 0, s->block_len * sizeof(float));
            ret = wma_run_level_decode(s, &s->gb, &s->coef_vlc[tindex],
                                       s->level_table[tindex], s->run_table[tindex],
                                       ptr, nb_coefs[ch],
                                       s->block_len, s->frame_len_bits, coef_nb_bits);
            if (ret < 0)
                return ret;
        }
    }

    {
        int n4 = s->block_len / 2;
        mdct_norm = 1.0 / (float)n4;
    }

    for (ch = 0; ch < channels; ch++) {
        if (s->channel_coded[ch]) {
            float *coefs1;
            float *coefs, *exponents, mult, mult1, noise;
            int i, j, nn, n1, last_high_band, esize;
            float exp_power[HIGH_BAND_MAX_SIZE];

            coefs1 = s->coefs1[ch];
            exponents = s->exponents[ch];
            esize = s->exponents_bsize[ch];
            mult = (float)pow(10.0, total_gain * 0.05) / s->max_exponent[ch];
            mult *= mdct_norm;
            coefs = s->coefs[ch];
            if (s->use_noise_coding) {
                mult1 = mult;
                for (i = 0; i < s->coefs_start; i++) {
                    *coefs++ = s->noise_table[s->noise_index] *
                               exponents[i << bsize >> esize] * mult1;
                    s->noise_index = (s->noise_index + 1) & (NOISE_TAB_SIZE - 1);
                }

                n1 = s->exponent_high_sizes[bsize];

                exponents = s->exponents[ch] +
                            (s->high_band_start[bsize] << bsize >> esize);
                last_high_band = 0;
                for (j = 0; j < n1; j++) {
                    nn = s->exponent_high_bands[s->frame_len_bits -
                                                s->block_len_bits][j];
                    if (s->high_band_coded[ch][j]) {
                        float e2, v2;
                        e2 = 0;
                        for (i = 0; i < nn; i++) {
                            v2 = exponents[i << bsize >> esize];
                            e2 += v2 * v2;
                        }
                        exp_power[j] = e2 / nn;
                        last_high_band = j;
                    }
                    exponents += nn << bsize >> esize;
                }

                exponents = s->exponents[ch] + (s->coefs_start << bsize >> esize);
                for (j = -1; j < n1; j++) {
                    if (j < 0)
                        nn = s->high_band_start[bsize] - s->coefs_start;
                    else
                        nn = s->exponent_high_bands[s->frame_len_bits -
                                                    s->block_len_bits][j];
                    if (j >= 0 && s->high_band_coded[ch][j]) {
                        mult1 = sqrtf(exp_power[j] / exp_power[last_high_band]);
                        mult1 = mult1 * (float)pow(10.0, s->high_band_values[ch][j] * 0.05);
                        mult1 = mult1 / (s->max_exponent[ch] * s->noise_mult);
                        mult1 *= mdct_norm;
                        for (i = 0; i < nn; i++) {
                            noise = s->noise_table[s->noise_index];
                            s->noise_index = (s->noise_index + 1) & (NOISE_TAB_SIZE - 1);
                            *coefs++ = noise * exponents[i << bsize >> esize] * mult1;
                        }
                        exponents += nn << bsize >> esize;
                    } else {
                        for (i = 0; i < nn; i++) {
                            noise = s->noise_table[s->noise_index];
                            s->noise_index = (s->noise_index + 1) & (NOISE_TAB_SIZE - 1);
                            *coefs++ = ((*coefs1++) + noise) *
                                       exponents[i << bsize >> esize] * mult;
                        }
                        exponents += nn << bsize >> esize;
                    }
                }

                nn = s->block_len - s->coefs_end[bsize];
                mult1 = mult * exponents[(-(1 << bsize)) >> esize];
                for (i = 0; i < nn; i++) {
                    *coefs++ = s->noise_table[s->noise_index] * mult1;
                    s->noise_index = (s->noise_index + 1) & (NOISE_TAB_SIZE - 1);
                }
            } else {
                for (i = 0; i < s->coefs_start; i++)
                    *coefs++ = 0.0;
                nn = nb_coefs[ch];
                for (i = 0; i < nn; i++)
                    *coefs++ = coefs1[i] * exponents[i << bsize >> esize] * mult;
                nn = s->block_len - s->coefs_end[bsize];
                for (i = 0; i < nn; i++)
                    *coefs++ = 0.0;
            }
        }
    }

    if (s->ms_stereo && s->channel_coded[1]) {
        if (!s->channel_coded[0]) {
            memset(s->coefs[0], 0, sizeof(float) * s->block_len);
            s->channel_coded[0] = 1;
        }

        for (int i = 0; i < s->block_len; i++) {
            float t = s->coefs[0][i] - s->coefs[1][i];
            s->coefs[0][i] += s->coefs[1][i];
            s->coefs[1][i] = t;
        }
    }

next:
    for (ch = 0; ch < channels; ch++) {
        int n4, index;

        n4 = s->block_len / 2;
        if (s->channel_coded[ch])
            mdct_run(&s->mdct[bsize], s->output, s->coefs[ch]);
        else if (!(s->ms_stereo && ch == 1))
            memset(s->output, 0, sizeof(s->output));

        index = (s->frame_len / 2) + s->block_pos - n4;
        wma_window(s, &s->frame_out[ch][index]);
    }

    s->block_num++;
    s->block_pos += s->block_len;
    if (s->block_pos >= s->frame_len)
        return 1;
    else
        return 0;
}

static int wma_decode_frame(WMAContext *s, float **samples, int samples_offset)
{
    int ret, ch;

    s->block_num = 0;
    s->block_pos = 0;
    for (;;) {
        ret = wma_decode_block(s);
        if (ret < 0)
            return -1;
        if (ret)
            break;
    }

    for (ch = 0; ch < s->channels; ch++) {
        memcpy(samples[ch] + samples_offset, s->frame_out[ch],
               s->frame_len * sizeof(float));
        memmove(&s->frame_out[ch][0], &s->frame_out[ch][s->frame_len],
                s->frame_len * sizeof(float));
    }

    return 0;
}

static void pcm_append(WMAContext *s, float **samples, int off, int count)
{
    int ch;
    if (s->pcm_len + count > s->pcm_cap) {
        int ncap = (s->pcm_cap ? s->pcm_cap : 1 << 16);
        while (s->pcm_len + count > ncap)
            ncap *= 2;
        for (ch = 0; ch < s->channels; ch++) {
            s->pcm[ch] = (float *)realloc(s->pcm[ch], sizeof(float) * ncap);
        }
        s->pcm_cap = ncap;
    }
    for (ch = 0; ch < s->channels; ch++)
        memcpy(s->pcm[ch] + s->pcm_len, samples[ch] + off, sizeof(float) * count);
    s->pcm_len += count;
}

static int wma_decode_superframe(WMAContext *s, const uint8_t *buf, int buf_size)
{
    int nb_frames, bit_offset, i, pos, len;
    uint8_t *q;
    float *samples[MAX_CHANNELS];
    float *framebuf[MAX_CHANNELS];
    int samples_offset;

    if (buf_size == 0) {
        for (i = 0; i < s->channels; i++)
            framebuf[i] = s->frame_out[i];
        pcm_append(s, framebuf, 0, s->frame_len);
        s->last_superframe_len = 0;
        return 0;
    }
    if (buf_size < s->block_align)
        return -1;
    if (s->block_align)
        buf_size = s->block_align;

    br_init(&s->gb, buf, buf_size * 8);

    if (s->use_bit_reservoir) {
        br_skip(&s->gb, 4);
        nb_frames = br_get(&s->gb, 4) - (s->last_superframe_len <= 0);
        if (nb_frames <= 0) {
            int is_error = nb_frames < 0 || br_left(&s->gb) <= 8;
            if (is_error)
                return -1;

            if ((s->last_superframe_len + buf_size - 1) > MAX_CODED_SUPERFRAME_SIZE)
                goto fail;

            q = s->last_superframe + s->last_superframe_len;
            len = buf_size - 1;
            while (len > 0) {
                *q++ = br_get(&s->gb, 8);
                len--;
            }
            memset(q, 0, PAD);

            s->last_superframe_len += 8 * buf_size - 8;
            return buf_size;
        }
    } else
        nb_frames = 1;

    for (i = 0; i < s->channels; i++) {
        framebuf[i] = (float *)malloc(sizeof(float) * nb_frames * s->frame_len);
        samples[i] = framebuf[i];
    }
    samples_offset = 0;

    if (s->use_bit_reservoir) {
        bit_offset = br_get(&s->gb, s->byte_offset_bits + 3);
        if (bit_offset > br_left(&s->gb))
            goto fail_frames;

        if (s->last_superframe_len > 0) {
            if ((s->last_superframe_len + ((bit_offset + 7) >> 3)) >
                MAX_CODED_SUPERFRAME_SIZE)
                goto fail_frames;
            q = s->last_superframe + s->last_superframe_len;
            len = bit_offset;
            while (len > 7) {
                *q++ = br_get(&s->gb, 8);
                len -= 8;
            }
            if (len > 0)
                *q++ = br_get(&s->gb, len) << (8 - len);
            memset(q, 0, PAD);

            br_init(&s->gb, s->last_superframe,
                    s->last_superframe_len * 8 + bit_offset);
            if (s->last_bitoffset > 0)
                br_skip(&s->gb, s->last_bitoffset);
            if (wma_decode_frame(s, samples, samples_offset) < 0)
                goto fail_frames;
            samples_offset += s->frame_len;
            nb_frames--;
        }

        pos = bit_offset + 4 + 4 + s->byte_offset_bits + 3;
        if (pos >= MAX_CODED_SUPERFRAME_SIZE * 8 || pos > buf_size * 8)
            goto fail_frames;
        br_init(&s->gb, buf + (pos >> 3), (buf_size - (pos >> 3)) * 8);
        len = pos & 7;
        if (len > 0)
            br_skip(&s->gb, len);

        s->reset_block_lengths = 1;
        for (i = 0; i < nb_frames; i++) {
            if (wma_decode_frame(s, samples, samples_offset) < 0)
                goto fail_frames;
            samples_offset += s->frame_len;
        }

        pos = br_count(&s->gb) +
              ((bit_offset + 4 + 4 + s->byte_offset_bits + 3) & ~7);
        s->last_bitoffset = pos & 7;
        pos >>= 3;
        len = buf_size - pos;
        if (len > MAX_CODED_SUPERFRAME_SIZE || len < 0)
            goto fail_frames;
        s->last_superframe_len = len;
        memcpy(s->last_superframe, buf + pos, len);
    } else {
        if (wma_decode_frame(s, samples, samples_offset) < 0)
            goto fail_frames;
        samples_offset += s->frame_len;
    }

    pcm_append(s, samples, 0, samples_offset);
    for (i = 0; i < s->channels; i++)
        free(framebuf[i]);
    return buf_size;

fail_frames:
    for (i = 0; i < s->channels; i++)
        free(framebuf[i]);
fail:
    s->last_superframe_len = 0;
    return -1;
}

typedef struct {
    uint8_t *data;
    int size;
} FileBuf;

static FileBuf read_file(const char *path)
{
    FileBuf fb = { NULL, 0 };
    FILE *f = fopen(path, "rb");
    if (!f)
        return fb;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    fb.data = (uint8_t *)malloc(sz + PAD * 2);
    fb.size = (int)fread(fb.data, 1, sz, f);
    memset(fb.data + fb.size, 0, PAD * 2);
    fclose(f);
    return fb;
}

static uint32_t rl32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint16_t rl16(const uint8_t *p)
{
    return (uint16_t)(p[0] | (p[1] << 8));
}

static void wr32(uint8_t *p, uint32_t v)
{
    p[0] = v; p[1] = v >> 8; p[2] = v >> 16; p[3] = v >> 24;
}

static void wr16(uint8_t *p, uint16_t v)
{
    p[0] = v; p[1] = v >> 8;
}

static int write_raw_f32(const char *path, float **pcm, int channels, int nsamples)
{
    FILE *f = fopen(path, "wb");
    if (!f)
        return -1;
    for (int i = 0; i < nsamples; i++)
        for (int ch = 0; ch < channels; ch++)
            fwrite(&pcm[ch][i], sizeof(float), 1, f);
    fclose(f);
    return 0;
}

static int write_wav(const char *path, float **pcm, int channels, int sample_rate, int nsamples)
{
    FILE *f = fopen(path, "wb");
    if (!f)
        return -1;
    int data_bytes = nsamples * channels * 2;
    uint8_t hdr[44];
    memcpy(hdr, "RIFF", 4);
    wr32(hdr + 4, 36 + data_bytes);
    memcpy(hdr + 8, "WAVEfmt ", 8);
    wr32(hdr + 16, 16);
    wr16(hdr + 20, 1);
    wr16(hdr + 22, channels);
    wr32(hdr + 24, sample_rate);
    wr32(hdr + 28, sample_rate * channels * 2);
    wr16(hdr + 32, channels * 2);
    wr16(hdr + 34, 16);
    memcpy(hdr + 36, "data", 4);
    wr32(hdr + 40, data_bytes);
    fwrite(hdr, 1, 44, f);

    int blk = 4096;
    int16_t *tmp = (int16_t *)malloc(sizeof(int16_t) * blk * channels);
    for (int i = 0; i < nsamples; i += blk) {
        int cnt = nsamples - i;
        if (cnt > blk)
            cnt = blk;
        for (int j = 0; j < cnt; j++) {
            for (int ch = 0; ch < channels; ch++) {
                float v = pcm[ch][i + j] * 32768.0f;
                int iv = (int)(v + (v >= 0 ? 0.5f : -0.5f));
                if (iv > 32767) iv = 32767;
                if (iv < -32768) iv = -32768;
                tmp[j * channels + ch] = (int16_t)iv;
            }
        }
        fwrite(tmp, sizeof(int16_t), cnt * channels, f);
    }
    free(tmp);
    fclose(f);
    return 0;
}

static int decode_xwma(const uint8_t *data, int size, const char *outpath, int verbose)
{
    if (size < 12 || memcmp(data, "RIFF", 4) || memcmp(data + 8, "XWMA", 4)) {
        fprintf(stderr, "not an xwma file\n");
        return -1;
    }

    int fmt_tag = 0, channels = 0, sample_rate = 0, byte_rate = 0, block_align = 0, bits = 0;
    uint8_t extra[64];
    int extra_size = 0;
    uint32_t *dpds = NULL;
    int dpds_n = 0;
    const uint8_t *wave = NULL;
    int wave_size = 0;

    int off = 12;
    while (off + 8 <= size) {
        uint32_t cid = rl32(data + off);
        uint32_t csz = rl32(data + off + 4);
        if (off + 8 + (int)csz > size)
            csz = size - off - 8;
        if (cid == 0x20746d66) {
            const uint8_t *p = data + off + 8;
            fmt_tag = rl16(p);
            channels = rl16(p + 2);
            sample_rate = rl32(p + 4);
            byte_rate = rl32(p + 8);
            block_align = rl16(p + 12);
            bits = rl16(p + 14);
            if (csz > 18) {
                int cb = rl16(p + 16);
                extra_size = cb;
                if (extra_size > (int)csz - 18)
                    extra_size = csz - 18;
                if (extra_size > 64)
                    extra_size = 64;
                memcpy(extra, p + 18, extra_size);
            }
        } else if (cid == 0x73647064) {
            dpds_n = csz / 4;
            dpds = (uint32_t *)malloc(sizeof(uint32_t) * (dpds_n + 1));
            for (int i = 0; i < dpds_n; i++)
                dpds[i] = rl32(data + off + 8 + i * 4);
        } else if (cid == 0x61746164) {
            wave = data + off + 8;
            wave_size = csz;
        }
        off += 8 + csz + (csz & 1);
    }

    if (!wave || !channels || !sample_rate || !block_align) {
        fprintf(stderr, "bad xwma header\n");
        free(dpds);
        return -1;
    }
    if (fmt_tag != 0x161) {
        fprintf(stderr, "unsupported codec tag 0x%x\n", fmt_tag);
        free(dpds);
        return -1;
    }

    int bit_rate = byte_rate * 8;

    if (channels == 1) {
        if (sample_rate == 22050 && (bit_rate == 48000 || bit_rate == 192000))
            bit_rate = 20000;
        else if (sample_rate == 32000 && (bit_rate == 48000 || bit_rate == 192000))
            bit_rate = 20000;
        else if (sample_rate == 44100 && (bit_rate == 96000 || bit_rate == 192000))
            bit_rate = 48000;
    } else if (channels == 2) {
        if (sample_rate == 22050 && (bit_rate == 48000 || bit_rate == 192000))
            bit_rate = 32000;
        else if (sample_rate == 32000 && bit_rate == 192000)
            bit_rate = 48000;
    }

    int flags2;
    if (extra_size >= 6)
        flags2 = rl16(extra + 4);
    else
        flags2 = 31;

    WMAContext *s = (WMAContext *)calloc(1, sizeof(WMAContext));
    if (wma_init(s, channels, sample_rate, bit_rate, block_align, flags2) < 0) {
        fprintf(stderr, "wma init failed\n");
        free(dpds);
        free(s);
        return -1;
    }

    int npk = (wave_size + block_align - 1) / block_align;
    for (int i = 0; i < npk; i++) {
        int off2 = i * block_align;
        int len = wave_size - off2;
        if (len > block_align)
            len = block_align;
        uint8_t pkt[MAX_CODED_SUPERFRAME_SIZE + PAD];
        memcpy(pkt, wave + off2, len);
        memset(pkt + len, 0, PAD);
        if (wma_decode_superframe(s, pkt, len) < 0) {
            fprintf(stderr, "packet %d decode error\n", i);
            break;
        }
    }
    wma_decode_superframe(s, NULL, 0);

    int skip = s->frame_len * 2;
    if (skip > s->pcm_len)
        skip = s->pcm_len;
    for (int ch = 0; ch < channels; ch++)
        memmove(s->pcm[ch], s->pcm[ch] + skip, sizeof(float) * (s->pcm_len - skip));
    s->pcm_len -= skip;

    if (dpds && dpds_n) {
        int expect = (int)(dpds[dpds_n - 1] / (channels * bits / 8));
        if (expect != s->pcm_len && verbose)
            fprintf(stderr, "dpds sample mismatch: %d vs %d\n", expect, s->pcm_len);
    }

    int rc;
    {
        const char *dot = strrchr(outpath, '.');
        if (dot && !strcmp(dot, ".f32"))
            rc = write_raw_f32(outpath, s->pcm, channels, s->pcm_len);
        else
            rc = write_wav(outpath, s->pcm, channels, sample_rate, s->pcm_len);
    }
    if (verbose)
        printf("%s: %d ch %d Hz %d samples\n", outpath, channels, sample_rate, s->pcm_len);

    wma_end(s);
    for (int ch = 0; ch < channels; ch++)
        free(s->pcm[ch]);
    free(s);
    free(dpds);
    return rc;
}

static int selftest(void)
{
    int fail = 0;
    for (int N = 8; N <= 64; N *= 2) {
        float *X = (float *)malloc(sizeof(float) * N);
        double *out = (double *)malloc(sizeof(double) * 2 * N);
        double *ref = (double *)malloc(sizeof(double) * 2 * N);
        double *H = (double *)malloc(sizeof(double) * N);
        for (int i = 0; i < N; i++)
            X[i] = (float)sin(i * 1.7) + (float)cos(i * 0.3);
        MDCT m;
        mdct_init(&m, N);
        mdct_run(&m, out, X);
        double s = 1.0 / 32768.0;
        double phase = M_PI / (4.0 * N);
        for (int i = 0; i < N / 2; i++) {
            double sd = 0, su = 0;
            double i_d = phase * (2 * N - 2 * i - 1);
            double i_u = phase * (3 * N + 2 * i + 1);
            for (int j = 0; j < N; j++) {
                sd += X[j] * cos((2 * j + 1) * i_d);
                su += X[j] * cos((2 * j + 1) * i_u);
            }
            H[i] = sd * s;
            H[N / 2 + i] = -su * s;
        }
        for (int i = 0; i < N / 2; i++)
            ref[i] = -H[N / 2 - 1 - i];
        for (int i = 0; i < N; i++)
            ref[N / 2 + i] = H[i];
        for (int i = 0; i < N / 2; i++)
            ref[2 * N - 1 - i] = H[N / 2 + i];
        double err = 0, sig = 0;
        for (int j = 0; j < 2 * N; j++) {
            err += (out[j] - ref[j]) * (out[j] - ref[j]);
            sig += ref[j] * ref[j];
        }
        printf("imdct N=%d relerr=%g\n", N, sqrt(err / sig));
        if (err / sig > 1e-18)
            fail = 1;
        mdct_free(&m);
        free(X); free(out); free(ref); free(H);
    }
    return fail;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s <input.xwma> [output.wav]\n", argv[0]);
        fprintf(stderr, "       %s <input_dir> [output_dir]\n", argv[0]);
        return 1;
    }
    if (!strcmp(argv[1], "--selftest"))
        return selftest();

    const char *in = argv[1];

    {
        struct stat st;
        if (stat(in, &st) == 0 && S_ISDIR(st.st_mode)) {
            const char *outdir = (argc > 2) ? argv[2] : ".";
            DIR *d = opendir(in);
            if (!d) {
                fprintf(stderr, "cannot open dir %s\n", in);
                return 1;
            }
            int fail = 0;
            struct dirent *e;
            while ((e = readdir(d)) != NULL) {
                size_t l = strlen(e->d_name);
                if (l < 5 || strcmp(e->d_name + l - 5, ".xwma"))
                    continue;
                char ipath[2048], opath[2048], base[1024];
                snprintf(ipath, sizeof(ipath), "%s/%s", in, e->d_name);
                snprintf(base, sizeof(base), "%s", e->d_name);
                base[l - 5] = 0;
                snprintf(opath, sizeof(opath), "%s/%s.wav", outdir, base);
                FileBuf fb = read_file(ipath);
                if (!fb.data) {
                    fprintf(stderr, "cannot read %s\n", ipath);
                    fail = 1;
                    continue;
                }
                if (decode_xwma(fb.data, fb.size, opath, 1) < 0)
                    fail = 1;
                free(fb.data);
            }
            closedir(d);
            return fail ? 1 : 0;
        }
    }

    const char *out = (argc > 2) ? argv[2] : NULL;
    char outbuf[1024];
    if (!out) {
        const char *dot = strrchr(in, '.');
        size_t n = dot ? (size_t)(dot - in) : strlen(in);
        snprintf(outbuf, sizeof(outbuf), "%.*s.wav", (int)n, in);
        out = outbuf;
    }

    FileBuf fb = read_file(in);
    if (!fb.data) {
        fprintf(stderr, "cannot read %s\n", in);
        return 1;
    }
    int rc = decode_xwma(fb.data, fb.size, out, 1);
    free(fb.data);
    return rc < 0 ? 1 : 0;
}
