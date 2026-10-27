#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

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

typedef struct {
    int n;
    int max_level;
    const uint32_t *huffcodes;
    const uint8_t *huffbits;
    const uint16_t *levels;
} CoefVLCTable;

typedef struct {
    int n;
    double *twr;
    double *twi;
    double *thr;
    double *thi;
    double *fr;
    double *fi;
} MDCT;

#include "wma_tables.h"

typedef struct {
    uint8_t *buf;
    int cap;
    int pos;
    uint32_t acc;
    int nbits;
} BitWriter;

typedef struct {
    uint8_t *buf;
    int cap;
    int len;
} BitBuf;

typedef struct {
    int channels;
    int sample_rate;
    int bit_rate;
    int block_align;
    int frame_len;
    int frame_len_bits;
    int nb_block_sizes;
    int byte_offset_bits;
    int use_noise_coding;
    int exponent_sizes[BLOCK_NB_SIZES];
    uint16_t exponent_bands[BLOCK_NB_SIZES][25];
    int coefs_start;
    int coefs_end[BLOCK_NB_SIZES];
    int exponent_high_sizes[BLOCK_NB_SIZES];
    int exponent_high_bands[BLOCK_NB_SIZES][HIGH_BAND_MAX_SIZE];
    int high_band_start[BLOCK_NB_SIZES];
    const CoefVLCTable *coef_vlcs[2];
    uint16_t *run_table[2];
    float *level_table[2];
    uint16_t *int_table[2];
    MDCT mdct;
    float *win;
    double *mdct_in;
    double *mdct_out;
    float *coefs[MAX_CHANNELS];
    float *prev_win[MAX_CHANNELS];
    int exp_idx[MAX_CHANNELS][25];
    int exp_nbands;
} EncState;

static void die(const char *m)
{
    fprintf(stderr, "%s\n", m);
    exit(1);
}

static void *xmalloc(size_t n)
{
    void *p = malloc(n);
    if (!p)
        die("oom");
    return p;
}

static int av_log2(unsigned int v)
{
    return 31 - __builtin_clz(v);
}

static void bw_init(BitWriter *b, uint8_t *buf, int cap)
{
    b->buf = buf;
    b->cap = cap;
    b->pos = 0;
    b->acc = 0;
    b->nbits = 0;
}

static void bw_put(BitWriter *b, int n, uint32_t v)
{
    b->acc = (b->acc << n) | (v & (n >= 32 ? 0xFFFFFFFFu : ((1u << n) - 1)));
    b->nbits += n;
    while (b->nbits >= 8) {
        b->nbits -= 8;
        if (b->pos < b->cap)
            b->buf[b->pos++] = (uint8_t)(b->acc >> b->nbits);
    }
}

static void bw_flush(BitWriter *b)
{
    while (b->nbits > 0) {
        b->nbits -= 8;
        if (b->pos < b->cap)
            b->buf[b->pos++] = (uint8_t)(b->acc << -b->nbits);
    }
    b->acc = 0;
}

static int bw_bits(BitWriter *b)
{
    return b->pos * 8 + b->nbits;
}

static void bb_init(BitBuf *b)
{
    b->cap = 1 << 16;
    b->buf = xmalloc(b->cap);
    b->len = 0;
}

static void bb_put(BitBuf *b, int n, uint32_t v)
{
    for (int i = n - 1; i >= 0; i--) {
        if (b->len >= b->cap * 8) {
            b->cap *= 2;
            b->buf = realloc(b->buf, b->cap);
        }
        int bit = (v >> i) & 1;
        int p = b->len >> 3;
        if (bit)
            b->buf[p] |= 1 << (7 - (b->len & 7));
        else
            b->buf[p] &= ~(1 << (7 - (b->len & 7)));
        b->len++;
    }
}

static void bb_cat(BitBuf *b, const uint8_t *data, int bits)
{
    for (int i = 0; i < bits; i++) {
        int bit = (data[i >> 3] >> (7 - (i & 7))) & 1;
        bb_put(b, 1, bit);
    }
}

static void bb_free(BitBuf *b)
{
    free(b->buf);
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
    int len = n * 2;
    m->twr = xmalloc(sizeof(double) * len);
    m->twi = xmalloc(sizeof(double) * len);
    m->thr = xmalloc(sizeof(double) * n);
    m->thi = xmalloc(sizeof(double) * n);
    m->fr = xmalloc(sizeof(double) * len);
    m->fi = xmalloc(sizeof(double) * len);
    for (int i = 0; i < len; i++) {
        double a = M_PI * (i + 0.5) / (2.0 * n);
        m->twr[i] = cos(a);
        m->twi[i] = sin(a);
    }
    for (int k = 0; k < n; k++) {
        double a = M_PI / 2.0 * (k + 0.5) + M_PI * k / (2.0 * n);
        m->thr[k] = cos(a);
        m->thi[k] = sin(a);
    }
}

static void mdct_free(MDCT *m)
{
    free(m->twr); free(m->twi); free(m->thr); free(m->thi);
    free(m->fr); free(m->fi);
}

static void mdct_forward(MDCT *m, double *out, const float *in)
{
    int n = m->n;
    int len = n * 2;
    double *fr = m->fr, *fi = m->fi;
    for (int i = 0; i < len; i++) {
        fr[i] = in[i] * m->twr[i];
        fi[i] = in[i] * m->twi[i];
    }
    fft(fr, fi, len, 1);
    for (int k = 0; k < n; k++)
        out[k] = fr[k] * m->thr[k] - fi[k] * m->thi[k];
}

static int wma_total_gain_to_bits_val(int total_gain)
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

static void init_coef_tables(EncState *s)
{
    for (int t = 0; t < 2; t++) {
        const CoefVLCTable *vt = s->coef_vlcs[t];
        int n = vt->n;
        uint16_t *run_table = xmalloc(sizeof(uint16_t) * n);
        float *flevel_table = xmalloc(sizeof(float) * n);
        uint16_t *int_table = xmalloc(sizeof(uint16_t) * n);
        int i = 2, level = 1, k = 0;
        while (i < n) {
            int_table[k] = i;
            int l = vt->levels[k++];
            for (int j = 0; j < l; j++) {
                run_table[i] = j;
                flevel_table[i] = (float)level;
                i++;
            }
            level++;
        }
        s->run_table[t] = run_table;
        s->level_table[t] = flevel_table;
        s->int_table[t] = int_table;
    }
}

static void enc_init(EncState *s, int channels, int sample_rate, int bit_rate)
{
    memset(s, 0, sizeof(*s));
    s->channels = channels;
    s->sample_rate = sample_rate;
    s->bit_rate = bit_rate;

    s->frame_len_bits = 11;
    if (sample_rate <= 16000)
        s->frame_len_bits = 9;
    else if (sample_rate <= 22050)
        s->frame_len_bits = 10;
    s->frame_len = 1 << s->frame_len_bits;
    s->nb_block_sizes = 1;

    s->block_align = (int)((int64_t)bit_rate * s->frame_len / ((int64_t)sample_rate * 8));
    if (s->block_align > MAX_CODED_SUPERFRAME_SIZE)
        s->block_align = MAX_CODED_SUPERFRAME_SIZE;
    if (s->block_align < 64)
        s->block_align = 64;

    int sample_rate1 = sample_rate;
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

    float bps = (float)bit_rate / (float)(channels * sample_rate);
    s->byte_offset_bits = av_log2((int)(bps * s->frame_len / 8.0 + 0.5)) + 2;

    float bps1 = bps;
    if (channels == 2)
        bps1 = bps * 1.6;
    s->use_noise_coding = 1;
    float high_freq = sample_rate * 0.5f;
    if (sample_rate1 == 44100) {
        if (bps1 >= 0.61f)
            s->use_noise_coding = 0;
        else
            high_freq *= 0.4f;
    } else if (sample_rate1 == 22050) {
        if (bps1 >= 1.16f)
            s->use_noise_coding = 0;
        else if (bps1 >= 0.72f)
            high_freq *= 0.7f;
        else
            high_freq *= 0.6f;
    } else if (sample_rate1 == 16000) {
        if (bps > 0.5f)
            high_freq *= 0.5f;
        else
            high_freq *= 0.3f;
    } else if (sample_rate1 == 11025)
        high_freq *= 0.7f;
    else if (sample_rate1 == 8000) {
        if (bps <= 0.625f)
            high_freq *= 0.5f;
        else if (bps > 0.75f)
            s->use_noise_coding = 0;
        else
            high_freq *= 0.65f;
    } else {
        if (bps >= 0.8f)
            high_freq *= 0.75f;
        else if (bps >= 0.6f)
            high_freq *= 0.6f;
        else
            high_freq *= 0.5f;
    }

    s->coefs_start = 0;
    for (int k = 0; k < s->nb_block_sizes; k++) {
        int block_len = s->frame_len >> k;
        const uint8_t *table = NULL;
        int a = s->frame_len_bits - BLOCK_MIN_BITS - k;
        if (a < 3) {
            if (sample_rate >= 44100)
                table = exponent_band_44100[a];
            else if (sample_rate >= 32000)
                table = exponent_band_32000[a];
            else if (sample_rate >= 22050)
                table = exponent_band_22050[a];
        }
        int n;
        if (table) {
            n = *table++;
            for (int i = 0; i < n; i++)
                s->exponent_bands[k][i] = table[i];
            s->exponent_sizes[k] = n;
        } else {
            int j = 0, lpos = 0;
            for (int i = 0; i < 25; i++) {
                int fa = ff_wma_critical_freqs[i];
                int pos = ((block_len * 2 * fa) + (sample_rate << 1)) / (4 * sample_rate);
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
        int j = 0, pos = 0;
        for (int i = 0; i < n; i++) {
            int start = pos;
            pos += s->exponent_bands[k][i];
            int end = pos;
            if (start < s->high_band_start[k])
                start = s->high_band_start[k];
            if (end > s->coefs_end[k])
                end = s->coefs_end[k];
            if (end > start)
                s->exponent_high_bands[k][j++] = end - start;
        }
        s->exponent_high_sizes[k] = j;
    }
    s->exp_nbands = s->exponent_sizes[0];

    float *w = xmalloc(sizeof(float) * s->frame_len);
    for (int i = 0; i < s->frame_len; i++)
        w[i] = (float)sin((i + 0.5) * M_PI / (2.0 * s->frame_len));
    s->win = w;

    mdct_init(&s->mdct, s->frame_len);
    s->mdct_in = xmalloc(sizeof(double) * s->frame_len * 2);
    s->mdct_out = xmalloc(sizeof(double) * s->frame_len);
    for (int ch = 0; ch < channels; ch++) {
        s->coefs[ch] = xmalloc(sizeof(float) * s->frame_len);
        s->prev_win[ch] = xmalloc(sizeof(float) * s->frame_len);
        memset(s->prev_win[ch], 0, sizeof(float) * s->frame_len);
    }

    int coef_vlc_table = 2;
    if (sample_rate >= 32000) {
        if (bps1 < 0.72f)
            coef_vlc_table = 0;
        else if (bps1 < 1.16f)
            coef_vlc_table = 1;
    }
    s->coef_vlcs[0] = &coef_vlcs[coef_vlc_table * 2];
    s->coef_vlcs[1] = &coef_vlcs[coef_vlc_table * 2 + 1];
    init_coef_tables(s);
}

static void apply_window_and_mdct(EncState *s, float *const *audio, int off, int avail)
{
    int N = s->frame_len;
    double n_scale = 2.0 * 32768.0 / N;
    for (int ch = 0; ch < s->channels; ch++) {
        float cur[MAX_CHANNELS == 0 ? 1 : BLOCK_MAX_SIZE];
        for (int i = 0; i < N; i++) {
            float v = 0;
            if (i < avail)
                v = audio[ch][off + i];
            cur[i] = (float)(v * n_scale);
        }
        float *in = (float *)s->mdct_in;
        for (int i = 0; i < N; i++)
            in[i] = s->prev_win[ch][i] * s->win[i];
        for (int i = 0; i < N; i++)
            in[N + i] = cur[i] * s->win[N - 1 - i];
        memcpy(s->prev_win[ch], cur, sizeof(float) * N);
        mdct_forward(&s->mdct, s->mdct_out, in);
        for (int i = 0; i < N; i++)
            s->coefs[ch][i] = (float)s->mdct_out[i];
    }
}

static void choose_exponents(EncState *s, int ch)
{
    const uint16_t *bands = s->exponent_bands[0];
    int nb = s->exp_nbands;
    int pos = 0;
    int last = 36;
    for (int b = 0; b < nb; b++) {
        float m = 0;
        int n = bands[b];
        for (int i = 0; i < n; i++) {
            float v = fabsf(s->coefs[ch][pos + i]);
            if (v > m)
                m = v;
        }
        pos += n;
        int idx;
        if (m <= 0)
            idx = 20;
        else {
            idx = (int)lrintf(16.0f * log10f(m));
            if (idx < -60)
                idx = -60;
            if (idx > 95)
                idx = 95;
        }
        if (idx > last + 60)
            idx = last + 60;
        if (idx < last - 60)
            idx = last - 60;
        s->exp_idx[ch][b] = idx;
        last = idx;
    }
}

static int encode_frame_body(EncState *s, BitWriter *bw, int total_gain, int min_bits)
{
    int channels = s->channels;
    int N = s->frame_len;
    int num_coefs = s->coefs_end[0] - s->coefs_start;
    int coef_nb_bits;
    float mult = (float)pow(10.0, total_gain * 0.05);
    int level[MAX_CHANNELS][BLOCK_MAX_SIZE];
    int fail = 0;

    for (int ch = 0; ch < channels; ch++) {
        float max_exp = 0;
        for (int b = 0; b < s->exp_nbands; b++) {
            float v = pow_tab[s->exp_idx[ch][b] + 60];
            if (v > max_exp)
                max_exp = v;
        }
        float m = mult / max_exp;
        m *= 2.0f / N;
        const uint16_t *bands = s->exponent_bands[0];
        int pos = 0;
        for (int b = 0; b < s->exp_nbands; b++) {
            float ev = pow_tab[s->exp_idx[ch][b] + 60];
            float scale = ev * m;
            int n = bands[b];
            for (int i = 0; i < n && pos < num_coefs; i++, pos++) {
                float t = s->coefs[ch][pos] / scale;
                long lv = lrintf(t);
                if (lv > 2000000000L)
                    lv = 2000000000L;
                if (lv < -2000000000L)
                    lv = -2000000000L;
                level[ch][pos] = (int)lv;
            }
        }
        for (int i = pos; i < N; i++)
            level[ch][i] = 0;
        int max_lvl = 1 << wma_total_gain_to_bits_val(total_gain);
        for (int i = 0; i < num_coefs; i++) {
            int a = level[ch][i];
            if (a < 0)
                a = -a;
            if (a >= max_lvl)
                fail = 1;
        }
    }
    if (fail)
        return -1;

    if (channels == 2)
        bw_put(bw, 1, 1);
    for (int ch = 0; ch < channels; ch++)
        bw_put(bw, 1, 1);

    {
        int v = total_gain - 1;
        while (v >= 127) {
            bw_put(bw, 7, 127);
            v -= 127;
        }
        bw_put(bw, 7, v);
    }

    coef_nb_bits = wma_total_gain_to_bits_val(total_gain);

    if (s->use_noise_coding) {
        for (int ch = 0; ch < channels; ch++)
            for (int i = 0; i < s->exponent_high_sizes[0]; i++)
                bw_put(bw, 1, 0);
    }

    for (int ch = 0; ch < channels; ch++) {
        int last = 36;
        for (int b = 0; b < s->exp_nbands; b++) {
            int code = s->exp_idx[ch][b] - last + 60;
            bw_put(bw, ff_aac_scalefactor_bits[code], ff_aac_scalefactor_code[code]);
            last = s->exp_idx[ch][b];
        }
    }

    for (int ch = 0; ch < channels; ch++) {
        int tindex = (ch == 1 && channels == 2);
        const CoefVLCTable *vt = s->coef_vlcs[tindex];
        const uint16_t *int_table = s->int_table[tindex];
        const uint16_t *levels = vt->levels;
        int run = 0;
        int covered = 0;
        for (int i = 0; i < num_coefs; i++) {
            int lv = level[ch][i];
            if (lv) {
                int alv = lv < 0 ? -lv : lv;
                int code = 0;
                if (alv <= vt->max_level && run < levels[alv - 1])
                    code = run + int_table[alv - 1];
                bw_put(bw, vt->huffbits[code], vt->huffcodes[code]);
                if (code == 0) {
                    bw_put(bw, coef_nb_bits, alv);
                    bw_put(bw, s->frame_len_bits, run);
                }
                bw_put(bw, 1, lv < 0);
                run = 0;
                covered = i + 1;
            } else
                run++;
        }
        if (min_bits > 0 && ch == channels - 1) {
            int esc = 1 + coef_nb_bits + s->frame_len_bits + 1;
            int cap = num_coefs - covered - 1;
            int need = (min_bits - bw_bits(bw) + esc - 1) / esc;
            if (need > cap)
                need = cap;
            if (need < 0)
                need = 0;
            for (int i = 0; i < need; i++) {
                bw_put(bw, vt->huffbits[0], vt->huffcodes[0]);
                bw_put(bw, coef_nb_bits, 0);
                bw_put(bw, s->frame_len_bits, 0);
                bw_put(bw, 1, 0);
            }
            covered += need;
        }
        if (covered < num_coefs)
            bw_put(bw, vt->huffbits[1], vt->huffcodes[1]);
    }
    return 0;
}

typedef struct {
    uint8_t buf[MAX_CODED_SUPERFRAME_SIZE];
    int bits;
} FrameBits;

static void encode_frame(EncState *s, float *const *audio, int off, int avail,
                         int budget_bits, FrameBits *out)
{
    apply_window_and_mdct(s, audio, off, avail);
    if (s->channels == 2) {
        for (int i = 0; i < s->frame_len; i++) {
            float a = s->coefs[0][i] * 0.5f;
            float b = s->coefs[1][i] * 0.5f;
            s->coefs[0][i] = a + b;
            s->coefs[1][i] = a - b;
        }
    }
    for (int ch = 0; ch < s->channels; ch++)
        choose_exponents(s, ch);

    BitWriter bw;
    static uint8_t tmp[MAX_CODED_SUPERFRAME_SIZE * 2];
    int lo = 1, hi = 2048, best = -1;
    while (lo <= hi) {
        int g = (lo + hi) / 2;
        bw_init(&bw, tmp, sizeof(tmp));
        if (encode_frame_body(s, &bw, g, 0) == 0) {
            int bits = bw_bits(&bw);
            if (bits <= budget_bits) {
                best = g;
                hi = g - 1;
            } else
                lo = g + 1;
        } else
            lo = g + 1;
    }
    if (best < 0)
        best = 2048;
    int min_bits = budget_bits / 8;
    if (min_bits < 96)
        min_bits = 96;
    bw_init(&bw, out->buf, sizeof(out->buf));
    if (encode_frame_body(s, &bw, best, min_bits) != 0)
        die("frame encode failed");
    bw_flush(&bw);
    out->bits = bw_bits(&bw);
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

typedef struct {
    float *pcm[MAX_CHANNELS];
    int channels;
    int sample_rate;
    int nsamples;
} AudioBuf;

static AudioBuf read_wav(const char *path)
{
    AudioBuf a;
    memset(&a, 0, sizeof(a));
    FILE *f = fopen(path, "rb");
    if (!f)
        die("cannot open input wav");
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *d = xmalloc(sz);
    if (fread(d, 1, sz, f) != (size_t)sz)
        die("read error");
    fclose(f);
    if (memcmp(d, "RIFF", 4) || memcmp(d + 8, "WAVE", 4))
        die("not a wav file");
    int fmt = 0, ch = 0, sr = 0, bits = 0;
    const uint8_t *pcm = NULL;
    int pcm_bytes = 0;
    int off = 12;
    while (off + 8 <= sz) {
        uint32_t id = rl32(d + off);
        uint32_t csz = rl32(d + off + 4);
        if (off + 8 + (int)csz > sz)
            csz = sz - off - 8;
        if (id == 0x20746d66) {
            fmt = rl16(d + off + 8);
            ch = rl16(d + off + 10);
            sr = rl32(d + off + 12);
            bits = rl16(d + off + 22);
        } else if (id == 0x61746164) {
            pcm = d + off + 8;
            pcm_bytes = csz;
        }
        off += 8 + csz + (csz & 1);
    }
    if (!pcm || ch < 1 || ch > 2)
        die("bad wav");
    if (fmt != 1 && fmt != 3)
        die("unsupported wav format");
    if (fmt == 1 && bits != 16 && bits != 24 && bits != 32)
        die("unsupported wav bit depth");
    if (fmt == 3 && bits != 32)
        die("unsupported float wav");
    int bytes_per = bits / 8;
    int frames = pcm_bytes / (bytes_per * ch);
    a.channels = ch;
    a.sample_rate = sr;
    a.nsamples = frames;
    for (int c = 0; c < ch; c++)
        a.pcm[c] = xmalloc(sizeof(float) * (frames + 1));
    for (int i = 0; i < frames; i++) {
        for (int c = 0; c < ch; c++) {
            const uint8_t *p = pcm + (i * ch + c) * bytes_per;
            float v;
            if (fmt == 3) {
                float t;
                memcpy(&t, p, 4);
                v = t;
            } else if (bits == 16) {
                int16_t t = (int16_t)rl16(p);
                v = t / 32768.0f;
            } else if (bits == 24) {
                int32_t t = (int32_t)((p[0] << 8) | (p[1] << 16) | (p[2] << 24));
                v = (t >> 8) / 8388608.0f;
            } else {
                int32_t t = (int32_t)rl32(p);
                v = t / 2147483648.0f;
            }
            a.pcm[c][i] = v;
        }
    }
    free(d);
    return a;
}

static void write_xwma(const char *path, EncState *s, const uint8_t *data, int data_bytes,
                       const uint32_t *dpds, int dpds_n)
{
    FILE *f = fopen(path, "wb");
    if (!f)
        die("cannot open output");
    int fmt_size = 18;
    int riff_size = 4 + (8 + fmt_size) + (8 + dpds_n * 4) + (8 + data_bytes);
    uint8_t hdr[64];
    memcpy(hdr, "RIFF", 4);
    wr32(hdr + 4, riff_size);
    memcpy(hdr + 8, "XWMA", 4);
    memcpy(hdr + 12, "fmt ", 4);
    wr32(hdr + 16, fmt_size);
    wr16(hdr + 20, 0x161);
    wr16(hdr + 22, s->channels);
    wr32(hdr + 24, s->sample_rate);
    wr32(hdr + 28, s->bit_rate / 8);
    wr16(hdr + 32, s->block_align);
    wr16(hdr + 34, 16);
    wr16(hdr + 36, 0);
    fwrite(hdr, 1, 38, f);
    uint8_t tmp[8];
    memcpy(tmp, "dpds", 4);
    wr32(tmp + 4, dpds_n * 4);
    fwrite(tmp, 1, 8, f);
    for (int i = 0; i < dpds_n; i++) {
        wr32(tmp, dpds[i]);
        fwrite(tmp, 1, 4, f);
    }
    memcpy(tmp, "data", 4);
    wr32(tmp + 4, data_bytes);
    fwrite(tmp, 1, 8, f);
    fwrite(data, 1, data_bytes, f);
    fclose(f);
}

static void encode_file(const char *in_path, const char *out_path, int bit_rate)
{
    AudioBuf audio = read_wav(in_path);
    EncState *s = xmalloc(sizeof(EncState));
    enc_init(s, audio.channels, audio.sample_rate, bit_rate);
    int N = s->frame_len;

    int pad_in = N;
    int total_in = pad_in + audio.nsamples + 2 * N;
    float *ext[MAX_CHANNELS];
    for (int ch = 0; ch < audio.channels; ch++) {
        ext[ch] = xmalloc(sizeof(float) * total_in);
        memset(ext[ch], 0, sizeof(float) * total_in);
        memcpy(ext[ch] + pad_in, audio.pcm[ch], sizeof(float) * audio.nsamples);
    }

    int nframes = (total_in + N - 1) / N + 1;
    FrameBits *frames = xmalloc(sizeof(FrameBits) * nframes);
    int hb = 4 + 4 + s->byte_offset_bits + 3;
    int budget = s->block_align * 8 - hb - 12;
    int fi;
    for (fi = 0; fi < nframes; fi++) {
        int off = fi * N;
        if (off >= total_in && fi > 0)
            break;
        int avail = total_in - off;
        if (avail < 0)
            avail = 0;
        if (avail > N)
            avail = N;
        encode_frame(s, ext, off, avail, budget, &frames[fi]);
    }
    nframes = fi;

    int pkt_bits = s->block_align * 8;
    int max_tail = (1 << (s->byte_offset_bits + 3)) - 1;

    int max_pkts = nframes * 3 + 8;
    uint8_t *data = xmalloc((size_t)max_pkts * s->block_align + 64);
    uint32_t *dpds = xmalloc(sizeof(uint32_t) * (max_pkts + 2));
    uint8_t *pkt = xmalloc(s->block_align + 64);

    BitBuf pending;
    bb_init(&pending);

    int npk = 0;
    int decoded_total = 0;
    int cur = 0;

    for (;;) {
        BitBuf payload;
        bb_init(&payload);
        int cont = pending.len > 0;
        int tail_bits = 0;
        if (cont) {
            bb_cat(&payload, pending.buf, pending.len);
            tail_bits = pending.len;
            pending.len = 0;
        }

        int complete = 0;
        int begun = 0;
        int done_all = 0;

        while (cur < nframes) {
            FrameBits *fb = &frames[cur];
            int pre = begun ? 3 : 9;
            int total = pre + fb->bits;
            int room = pkt_bits - hb - payload.len;
            if (total <= room && complete < 14) {
                bb_put(&payload, pre, 0);
                bb_cat(&payload, fb->buf, fb->bits);
                begun++;
                complete++;
                cur++;
                continue;
            }
            if (room >= 1) {
                for (int i = 0; i < room; i++)
                    bb_put(&payload, 1, i < pre ? 0 : (fb->buf[(i - pre) >> 3] >> (7 - ((i - pre) & 7))) & 1);
                for (int i = room; i < total; i++)
                    bb_put(&pending, 1, i < pre ? 0 : (fb->buf[(i - pre) >> 3] >> (7 - ((i - pre) & 7))) & 1);
                int tail = total - room;
                if (tail > max_tail)
                    die("frame tail exceeds bit_offset field");
                begun++;
                cur++;
            }
            break;
        }
        if (cur >= nframes)
            done_all = 1;

        int field = complete + 1;
        if (field > 15)
            die("too many frames per packet");

        memset(pkt, 0, s->block_align);
        BitWriter hw;
        bw_init(&hw, pkt, s->block_align);
        bw_put(&hw, 4, npk & 15);
        bw_put(&hw, 4, field);
        bw_put(&hw, s->byte_offset_bits + 3, tail_bits);
        for (int i = 0; i < payload.len; i++)
            bw_put(&hw, 1, (payload.buf[i >> 3] >> (7 - (i & 7))) & 1);
        bw_flush(&hw);
        memcpy(data + (size_t)npk * s->block_align, pkt, s->block_align);

        decoded_total += field - (cont ? 0 : 1);
        dpds[npk] = (uint32_t)((decoded_total > 1 ? decoded_total - 1 : 0) * (int64_t)N * s->channels * 2);
        npk++;
        bb_free(&payload);

        if (done_all && pending.len == 0)
            break;
    }

    int data_bytes = npk * s->block_align;
    write_xwma(out_path, s, data, data_bytes, dpds, npk);
    printf("%s: %d ch %d Hz %d bps, %d frames, %d packets, %d bytes\n",
           out_path, s->channels, s->sample_rate, bit_rate, nframes, npk, data_bytes);
}

static int selftest(void)
{
    int fail = 0;
    for (int N = 8; N <= 64; N *= 2) {
        float *x = xmalloc(sizeof(float) * 2 * N);
        double *out = xmalloc(sizeof(double) * N);
        double *ref = xmalloc(sizeof(double) * N);
        for (int i = 0; i < 2 * N; i++)
            x[i] = (float)sin(i * 1.3) + (float)cos(i * 0.7);
        MDCT m;
        mdct_init(&m, N);
        mdct_forward(&m, out, x);
        for (int k = 0; k < N; k++) {
            double acc = 0;
            for (int n = 0; n < 2 * N; n++)
                acc += x[n] * cos(M_PI / N * (n + 0.5 + N / 2.0) * (k + 0.5));
            ref[k] = acc;
        }
        double err = 0, sig = 0;
        for (int k = 0; k < N; k++) {
            err += (out[k] - ref[k]) * (out[k] - ref[k]);
            sig += ref[k] * ref[k];
        }
        printf("mdct fwd N=%d relerr=%g\n", N, sqrt(err / sig));
        if (err / sig > 1e-18)
            fail = 1;
        mdct_free(&m);
        free(x); free(out); free(ref);
    }
    return fail;
}

int main(int argc, char **argv)
{
    if (argc >= 2 && !strcmp(argv[1], "--selftest"))
        return selftest();
    if (argc < 3) {
        fprintf(stderr, "usage: %s input.wav output.xwma [bitrate]\n", argv[0]);
        return 1;
    }
    int bitrate = (argc > 3) ? atoi(argv[3]) : 96000;
    if (bitrate < 8000)
        die("bitrate too low");
    encode_file(argv[1], argv[2], bitrate);
    return 0;
}
