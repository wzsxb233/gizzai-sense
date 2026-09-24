#include "series.h"

#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstring>

// Chronos masks with finfo.min; anything this negative is zero after the softmax and, unlike -inf,
// still behaves when a whole row is masked.
static const float NEG = -1e30f;

struct series_model {
    ggml_context *     ctx_w = nullptr;
    gguf_context *     gguf  = nullptr;
    series_hparams     hp;
    std::vector<float> quantiles;
    int                n_threads = 4;
};

// Read a number whatever width the writer chose. The converter takes these straight from Chronos'
// config.json, where 8192 arrives as an int and 1e-6 as a float, so asking for one fixed type aborts.
static double kv_num(const gguf_context * g, const char * key, double def) {
    const int64_t i = gguf_find_key(g, key);
    if (i < 0) return def;
    switch (gguf_get_kv_type(g, i)) {
        case GGUF_TYPE_UINT8:   return gguf_get_val_u8(g, i);
        case GGUF_TYPE_INT8:    return gguf_get_val_i8(g, i);
        case GGUF_TYPE_UINT16:  return gguf_get_val_u16(g, i);
        case GGUF_TYPE_INT16:   return gguf_get_val_i16(g, i);
        case GGUF_TYPE_UINT32:  return gguf_get_val_u32(g, i);
        case GGUF_TYPE_INT32:   return gguf_get_val_i32(g, i);
        case GGUF_TYPE_UINT64:  return (double) gguf_get_val_u64(g, i);
        case GGUF_TYPE_INT64:   return (double) gguf_get_val_i64(g, i);
        case GGUF_TYPE_FLOAT32: return gguf_get_val_f32(g, i);
        case GGUF_TYPE_FLOAT64: return gguf_get_val_f64(g, i);
        case GGUF_TYPE_BOOL:    return gguf_get_val_bool(g, i) ? 1.0 : 0.0;
        default:
            fprintf(stderr, "[series] %s is not a number\n", key);
            return def;
    }
}

static int   kv_u32(const gguf_context * g, const char * key, int def)     { return (int)   kv_num(g, key, def); }
static float kv_f32(const gguf_context * g, const char * key, float def)   { return (float) kv_num(g, key, def); }

static ggml_tensor * T(series_model * m, const std::string & name) {
    ggml_tensor * t = ggml_get_tensor(m->ctx_w, name.c_str());
    if (!t) {
        fprintf(stderr, "[series] missing tensor: %s\n", name.c_str());
    }
    return t;
}

series_model * series_load(const char * path, int n_threads) {
    series_model * m = new series_model();
    gguf_init_params p = { /*no_alloc*/ false, /*ctx*/ &m->ctx_w };
    m->gguf = gguf_init_from_file(path, p);
    if (!m->gguf) {
        fprintf(stderr, "[series] cannot read %s\n", path);
        delete m;
        return nullptr;
    }
    m->n_threads = n_threads;
    series_hparams & h = m->hp;
    h.d_model             = kv_u32(m->gguf, "ts.d_model", 768);
    h.d_ff                = kv_u32(m->gguf, "ts.d_ff", 3072);
    h.n_heads             = kv_u32(m->gguf, "ts.n_heads", 12);
    h.d_kv                = kv_u32(m->gguf, "ts.d_kv", 64);
    h.n_layers            = kv_u32(m->gguf, "ts.n_layers", 12);
    h.patch_size          = kv_u32(m->gguf, "ts.patch_size", 16);
    h.output_patch_size   = kv_u32(m->gguf, "ts.output_patch_size", 16);
    h.context_length      = kv_u32(m->gguf, "ts.context_length", 8192);
    h.reg_token_id        = kv_u32(m->gguf, "ts.reg_token_id", 1);
    h.pad_token_id        = kv_u32(m->gguf, "ts.pad_token_id", 0);
    h.eps                 = kv_f32(m->gguf, "ts.layer_norm_epsilon", 1e-6f);
    h.rope_theta          = kv_f32(m->gguf, "ts.rope_theta", 10000.0f);
    h.time_encoding_scale = kv_f32(m->gguf, "ts.time_encoding_scale", (float) h.context_length);
    h.hidden              = kv_u32(m->gguf, "sense_t.hidden", 1536);
    const int64_t qi      = gguf_find_key(m->gguf, "ts.quantiles");
    h.n_quantiles         = qi < 0 ? 0 : (int) gguf_get_arr_n(m->gguf, qi);
    if (qi >= 0) {
        const float * q = (const float *) gguf_get_arr_data(m->gguf, qi);
        m->quantiles.assign(q, q + h.n_quantiles);
    }
    return m;
}

void series_free(series_model * m) {
    if (!m) return;
    if (m->gguf)  gguf_free(m->gguf);
    if (m->ctx_w) ggml_free(m->ctx_w);
    delete m;
}

const series_hparams & series_hp(const series_model * m) { return m->hp; }

const std::vector<float> & series_quantiles(const series_model * m) { return m->quantiles; }

// T5 norm: root mean square only, no mean subtraction and no bias.
static ggml_tensor * t5_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, float eps) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), w);
}

// out = output_layer(relu(hidden_layer(x))) + residual_layer(x)
static ggml_tensor * res_block(ggml_context * ctx, series_model * m, const std::string & p, ggml_tensor * x) {
    ggml_tensor * h = ggml_add(ctx, ggml_mul_mat(ctx, T(m, p + ".hidden.weight"), x), T(m, p + ".hidden.bias"));
    h = ggml_relu(ctx, h);
    ggml_tensor * o = ggml_add(ctx, ggml_mul_mat(ctx, T(m, p + ".out.weight"), h), T(m, p + ".out.bias"));
    ggml_tensor * r = ggml_add(ctx, ggml_mul_mat(ctx, T(m, p + ".res.weight"), x), T(m, p + ".res.bias"));
    return ggml_add(ctx, o, r);
}

// Multi-head attention over dimension 1 of x {d_model, n_tok, n_bat}.
// `pos` non-null applies RoPE to q and k (time attention); group attention passes null.
// The softmax scale is 1.0 on purpose: Chronos-2 does not divide by sqrt(d_kv).
static ggml_tensor * mha(ggml_context * ctx, series_model * m, const std::string & p, ggml_tensor * x,
                         ggml_tensor * mask, ggml_tensor * pos) {
    const series_hparams & h = m->hp;
    const int64_t n_tok = x->ne[1], n_bat = x->ne[2];

    ggml_tensor * q = ggml_reshape_4d(ctx, ggml_mul_mat(ctx, T(m, p + "_q.weight"), x), h.d_kv, h.n_heads, n_tok, n_bat);
    ggml_tensor * k = ggml_reshape_4d(ctx, ggml_mul_mat(ctx, T(m, p + "_k.weight"), x), h.d_kv, h.n_heads, n_tok, n_bat);
    ggml_tensor * v = ggml_reshape_4d(ctx, ggml_mul_mat(ctx, T(m, p + "_v.weight"), x), h.d_kv, h.n_heads, n_tok, n_bat);

    if (pos) {   // NEOX rotates (i, i + d/2), which is torch's rotate_half
        q = ggml_rope_ext(ctx, q, pos, nullptr, h.d_kv, GGML_ROPE_TYPE_NEOX, 0,
                          h.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(ctx, k, pos, nullptr, h.d_kv, GGML_ROPE_TYPE_NEOX, 0,
                          h.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    }
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));            // {d_kv, n_tok, n_heads, n_bat}
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    ggml_tensor * kq = ggml_mul_mat(ctx, k, q);                      // {kv, q, n_heads, n_bat}
    kq = ggml_soft_max_ext(ctx, kq, mask, 1.0f, 0.0f);               // scale 1.0: scores are unscaled

    ggml_tensor * vt  = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));   // {n_tok, d_kv, n_heads, n_bat}
    ggml_tensor * kqv = ggml_mul_mat(ctx, vt, kq);                          // {d_kv, n_tok, n_heads, n_bat}
    kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));               // {d_kv, n_heads, n_tok, n_bat}
    kqv = ggml_reshape_3d(ctx, kqv, h.d_model, n_tok, n_bat);
    return ggml_mul_mat(ctx, T(m, p + "_o.weight"), kqv);
}

static std::string two(int i) {
    char b[8];
    snprintf(b, sizeof(b), "%02d", i);
    return b;
}

static void fetch(ggml_tensor * t, std::vector<float> & dst) {
    dst.resize(ggml_nelements(t));
    memcpy(dst.data(), t->data, dst.size() * sizeof(float));
}

bool series_patch_embed(series_model * m, const float * patches, int n_rows, int n_patches,
                        std::vector<float> & out) {
    const int in_dim = m->hp.patch_size * 3;   // [time_encoding, values, mask]
    ggml_init_params ip = { 256ull << 20, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return false;

    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, in_dim, n_patches, n_rows);
    memcpy(x->data, patches, ggml_nbytes(x));
    ggml_tensor * y = res_block(ctx, m, "ts.patch_embed", x);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_graph_compute_with_ctx(ctx, gf, m->n_threads);
    fetch(y, out);
    ggml_free(ctx);
    return true;
}

bool series_prepare(const series_hparams & h, const float * rows, int n_rows, int n_steps,
                    const float * futures, int n_fut_steps, int n_fut_patches, series_batch & b) {
    const int P     = h.patch_size;
    const int n_ctx = (n_steps + P - 1) / P;
    const int pad   = n_ctx * P - n_steps;          // short rows are padded on the LEFT, and unobserved
    const int C     = n_ctx * P;
    const int n_pos = n_ctx + 1 + n_fut_patches;

    b.n_rows = n_rows; b.n_ctx = n_ctx; b.n_fut = n_fut_patches;
    b.patched_ctx.assign((size_t) n_rows * n_ctx * P * 3, 0.0f);
    b.patched_fut.assign((size_t) n_rows * n_fut_patches * P * 3, 0.0f);
    b.attn.assign((size_t) n_rows * n_pos, 1.0f);   // the register and the future are always attended
    b.loc.resize(n_rows);
    b.scale.resize(n_rows);

    for (int r = 0; r < n_rows; ++r) {
        const float * x = rows + (size_t) r * n_steps;

        // standardise over the row's observed values; an entirely missing row gets loc 0, scale 1
        double sum = 0.0;
        int    cnt = 0;
        for (int i = 0; i < n_steps; ++i) {
            if (!std::isnan(x[i])) { sum += x[i]; ++cnt; }
        }
        const double loc = cnt ? sum / cnt : 0.0;
        double var = 0.0;
        for (int i = 0; i < n_steps; ++i) {
            if (!std::isnan(x[i])) var += (x[i] - loc) * (x[i] - loc);
        }
        double scale = cnt ? std::sqrt(var / cnt) : 1.0;
        if (scale == 0.0) scale = 1e-5;             // Chronos' own floor for a flat row
        b.loc[r]   = (float) loc;
        b.scale[r] = (float) scale;

        for (int p = 0; p < n_ctx; ++p) {
            float * dst  = &b.patched_ctx[((size_t) r * n_ctx + p) * P * 3];
            int     live = 0;
            for (int j = 0; j < P; ++j) {
                const int  i  = p * P + j;          // index into the padded context
                const int  t  = i - pad;
                const bool ok = t >= 0 && !std::isnan(x[t]);
                dst[j]         = (float) (i - C) / h.time_encoding_scale;   // [-C, 0) / context_length
                dst[P + j]     = ok ? (float) std::asinh((x[t] - loc) / scale) : 0.0f;
                dst[2 * P + j] = ok ? 1.0f : 0.0f;
                live += ok ? 1 : 0;
            }
            b.attn[(size_t) r * n_pos + p] = live > 0 ? 1.0f : 0.0f;
        }
        for (int p = 0; p < n_fut_patches; ++p) {
            float * dst = &b.patched_fut[((size_t) r * n_fut_patches + p) * P * 3];
            for (int j = 0; j < P; ++j) {
                const int   t  = p * P + j;
                const float v  = (futures && t < n_fut_steps) ? futures[(size_t) r * n_fut_steps + t] : NAN;
                const bool  ok = !std::isnan(v);
                dst[j]         = (float) t / h.time_encoding_scale;         // [0, K*P) / context_length
                dst[P + j]     = ok ? (float) std::asinh((v - loc) / scale) : 0.0f;
                dst[2 * P + j] = ok ? 1.0f : 0.0f;
            }
        }
    }
    return true;
}

bool series_forecast_base(series_model * m, const float * states, int n_rows, int n_pos, int n_fut,
                          const float * loc, const float * scale, std::vector<float> & out) {
    const series_hparams & h = m->hp;
    const int Q = h.n_quantiles, P = h.output_patch_size;
    if (Q <= 0) {
        fprintf(stderr, "[series] this GGUF carries no quantile levels\n");
        return false;
    }
    ggml_init_params ip = { 256ull << 20, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return false;

    // only the future positions are forecast, and they sit at the end
    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, h.d_model, n_fut, n_rows);
    for (int r = 0; r < n_rows; ++r) {
        memcpy((float *) x->data + (size_t) r * n_fut * h.d_model,
               states + ((size_t) r * n_pos + (n_pos - n_fut)) * h.d_model,
               (size_t) n_fut * h.d_model * sizeof(float));
    }
    ggml_tensor * y = res_block(ctx, m, "ts.out_patch", x);      // {Q*P, n_fut, n_rows}

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_graph_compute_with_ctx(ctx, gf, m->n_threads);

    // "b n (q p) -> b q (n p)": the head emits every quantile for one patch, we want every step of one
    // quantile. Then undo the standardisation the input went through, arcsinh included.
    const float * src = (const float *) y->data;
    out.assign((size_t) n_rows * Q * n_fut * P, 0.0f);
    for (int r = 0; r < n_rows; ++r) {
        for (int n = 0; n < n_fut; ++n) {
            for (int q = 0; q < Q; ++q) {
                for (int p = 0; p < P; ++p) {
                    float v = src[((size_t) r * n_fut + n) * Q * P + (size_t) q * P + p];
                    if (loc && scale) v = std::sinh(v) * scale[r] + loc[r];
                    out[((size_t) r * Q + q) * n_fut * P + (size_t) n * P + p] = v;
                }
            }
        }
    }
    ggml_free(ctx);
    return true;
}

// One row of a weight, whatever type it was stored as.
static void row_to_f32(ggml_tensor * t, int row, std::vector<float> & out) {
    const int64_t n = t->ne[0];
    out.resize(n);
    const char * src = (const char *) t->data + (size_t) row * t->nb[1];
    if (t->type == GGML_TYPE_F32) {
        memcpy(out.data(), src, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) src, out.data(), n);
    } else {
        fprintf(stderr, "[series] %s is stored as %s, which this reader does not handle\n",
                ggml_get_name(t), ggml_type_name(t->type));
    }
}

bool series_soft_tokens(series_model * m, const float * rows, int n_rows, int n_steps,
                        const float * futures, int n_fut_steps, int n_target, int n_fut_patches,
                        std::vector<float> & out, std::vector<int32_t> & roles) {
    const series_hparams & h = m->hp;
    series_batch b;
    if (!series_prepare(h, rows, n_rows, n_steps, futures, n_fut_steps, n_fut_patches, b)) return false;

    std::vector<float> ctx_emb, fut_emb, reg;
    if (!series_patch_embed(m, b.patched_ctx.data(), n_rows, b.n_ctx, ctx_emb)) return false;
    if (!series_patch_embed(m, b.patched_fut.data(), n_rows, b.n_fut, fut_emb)) return false;
    row_to_f32(T(m, "ts.tok_embd.weight"), h.reg_token_id, reg);

    // [context patches, REG, future patches] per row, which is the order the encoder was trained on
    const int d = h.d_model, n_pos = b.n_ctx + 1 + b.n_fut;
    std::vector<float> embeds((size_t) n_rows * n_pos * d);
    for (int r = 0; r < n_rows; ++r) {
        float * dst = &embeds[(size_t) r * n_pos * d];
        memcpy(dst, &ctx_emb[(size_t) r * b.n_ctx * d], (size_t) b.n_ctx * d * sizeof(float));
        memcpy(dst + (size_t) b.n_ctx * d, reg.data(), (size_t) d * sizeof(float));
        memcpy(dst + (size_t) (b.n_ctx + 1) * d, &fut_emb[(size_t) r * b.n_fut * d],
               (size_t) b.n_fut * d * sizeof(float));
    }

    std::vector<float> gids(n_rows, 0.0f);          // one item: every row attends to every other
    std::vector<float> states;
    if (!series_encode(m, embeds.data(), gids.data(), b.attn.data(), n_rows, n_pos, states)) return false;

    roles.assign(n_pos, 0);
    roles[b.n_ctx] = 1;                             // register
    for (int p = 0; p < b.n_fut; ++p) roles[b.n_ctx + 1 + p] = 2;
    states.resize((size_t) n_target * n_pos * d);   // the target variates' rows lead the batch
    return series_project(m, states.data(), roles.data(), n_target, n_pos, out);
}

bool series_project(series_model * m, const float * states, const int32_t * roles,
                    int n_var, int n_pos, std::vector<float> & out) {
    const series_hparams & h = m->hp;
    ggml_init_params ip = { 256ull << 20, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return false;

    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, h.d_model, n_pos, n_var);
    memcpy(x->data, states, ggml_nbytes(x));
    ggml_tensor * ri = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_pos);
    memcpy(ri->data, roles, n_pos * sizeof(int32_t));
    ggml_tensor * vi = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_var);
    for (int v = 0; v < n_var; ++v) ((int32_t *) vi->data)[v] = v;

    // role varies down the sequence, variate across the rows; both broadcast over the other axis
    x = ggml_add(ctx, x, ggml_get_rows(ctx, T(m, "sense.role_embd.weight"), ri));
    x = ggml_add(ctx, x, ggml_reshape_3d(ctx, ggml_get_rows(ctx, T(m, "sense.variate_embd.weight"), vi),
                                         h.d_model, 1, n_var));

    // a real LayerNorm here, mean subtracted and biased -- not the T5 norm Chronos uses inside
    x = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, 1e-5f), T(m, "sense.proj_norm.weight")),
                 T(m, "sense.proj_norm.bias"));
    x = ggml_add(ctx, ggml_mul_mat(ctx, T(m, "sense.proj_fc1.weight"), x), T(m, "sense.proj_fc1.bias"));
    x = ggml_gelu_erf(ctx, x);                       // torch's nn.GELU is the erf form, not the tanh one
    x = ggml_add(ctx, ggml_mul_mat(ctx, T(m, "sense.proj_fc2.weight"), x), T(m, "sense.proj_fc2.bias"));

    // onto the sphere real token embeddings sit on: rms_norm scales by sqrt(n)/||x||, so divide it back
    const float log_scale = ((const float *) T(m, "sense.log_scale")->data)[0];
    x = ggml_scale(ctx, ggml_rms_norm(ctx, x, 1e-12f), expf(log_scale) / sqrtf((float) h.hidden));

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, x);
    ggml_graph_compute_with_ctx(ctx, gf, m->n_threads);
    fetch(x, out);
    ggml_free(ctx);
    return true;
}

bool series_encode(series_model * m, const float * embeds, const float * group_ids, const float * mask,
                   int n_rows, int n_pos, std::vector<float> & out,
                   std::map<std::string, std::vector<float>> * stages) {
    const series_hparams & h = m->hp;
    ggml_init_params ip = { 1024ull << 20, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return false;

    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, h.d_model, n_pos, n_rows);
    memcpy(x->data, embeds, ggml_nbytes(x));

    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_pos);
    for (int i = 0; i < n_pos; ++i) ((int32_t *) pos->data)[i] = i;   // plain arange, REG included

    bool all_on = true;
    for (int i = 0; i < n_rows * n_pos; ++i) {
        if (mask[i] < 0.5f) { all_on = false; break; }
    }
    ggml_tensor * tmask = nullptr;            // {kv, q, 1, rows}: a position hidden in time
    if (!all_on) {
        tmask = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_pos, n_pos, 1, n_rows);
        float * d = (float *) tmask->data;
        for (int r = 0; r < n_rows; ++r)
            for (int q = 0; q < n_pos; ++q)
                for (int k = 0; k < n_pos; ++k)
                    d[(size_t) r * n_pos * n_pos + (size_t) q * n_pos + k] =
                        mask[r * n_pos + k] > 0.5f ? 0.0f : NEG;
    }
    // {kv_row, q_row, 1, pos}: rows of one group, at a position where that row is live
    ggml_tensor * gmask = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_rows, n_rows, 1, n_pos);
    {
        float * d = (float *) gmask->data;
        for (int t = 0; t < n_pos; ++t)
            for (int q = 0; q < n_rows; ++q)
                for (int b = 0; b < n_rows; ++b)
                    d[(size_t) t * n_rows * n_rows + (size_t) q * n_rows + b] =
                        (group_ids[q] == group_ids[b] && mask[b * n_pos + t] > 0.5f) ? 0.0f : NEG;
    }

    std::vector<std::pair<std::string, ggml_tensor *>> keep;
    for (int il = 0; il < h.n_layers; ++il) {
        const std::string p = "ts.blk." + std::to_string(il) + ".";

        ggml_tensor * n1 = t5_norm(ctx, x, T(m, p + "time_norm.weight"), h.eps);
        x = ggml_add(ctx, x, mha(ctx, m, p + "time_attn", n1, tmask, pos));

        // group attention runs with time and batch flipped: attention across variates at one patch
        ggml_tensor * xt = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));      // {d, rows, pos}
        ggml_tensor * n2 = t5_norm(ctx, xt, T(m, p + "group_norm.weight"), h.eps);
        xt = ggml_add(ctx, xt, mha(ctx, m, p + "group_attn", n2, gmask, nullptr));
        x  = ggml_cont(ctx, ggml_permute(ctx, xt, 0, 2, 1, 3));                   // back to {d, pos, rows}

        ggml_tensor * n3 = t5_norm(ctx, x, T(m, p + "ffn_norm.weight"), h.eps);
        ggml_tensor * f  = ggml_relu(ctx, ggml_mul_mat(ctx, T(m, p + "ffn_up.weight"), n3));
        x = ggml_add(ctx, x, ggml_mul_mat(ctx, T(m, p + "ffn_down.weight"), f));
        if (stages) keep.emplace_back("block_" + two(il), x);
    }
    ggml_tensor * fin = t5_norm(ctx, x, T(m, "ts.output_norm.weight"), h.eps);
    if (stages) keep.emplace_back("final_norm", fin);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, fin);
    ggml_graph_compute_with_ctx(ctx, gf, m->n_threads);

    fetch(fin, out);
    if (stages) {
        for (auto & kv : keep) fetch(kv.second, (*stages)[kv.first]);
    }
    ggml_free(ctx);
    return true;
}
