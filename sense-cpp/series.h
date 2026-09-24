// Chronos-2's encoder as a ggml graph: the series side of Sense-T, on the engine llama.cpp already uses.
//
// This is the piece a runtime needs before a series can be a modality. It reads the GGUF written by
// scripts/convert_series_gguf.py and runs the encoder; the language model stays a stock Gemma 4 GGUF.
//
// Three things about Chronos-2 that a port gets wrong silently, all of them load-bearing here:
//   * attention scores are NOT divided by sqrt(d_kv) — the reference multiplies q·k and softmaxes it raw
//   * group attention transposes time and batch, attending across variates at one patch, without RoPE
//   * the norm is T5-style: no mean subtraction, no bias
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

struct series_hparams {
    int   d_model = 0, d_ff = 0, n_heads = 0, d_kv = 0, n_layers = 0;
    int   patch_size = 0, output_patch_size = 0, context_length = 0;
    int   reg_token_id = 1, pad_token_id = 0;
    float eps = 1e-6f, rope_theta = 10000.0f, time_encoding_scale = 8192.0f;
    int   n_quantiles = 0;
    int   hidden = 0;          // the language model's embedding width: what a soft token must be
};

struct series_model;

series_model *         series_load(const char * gguf_path, int n_threads = 4);
void                   series_free(series_model * m);
const series_hparams & series_hp(const series_model * m);
const std::vector<float> & series_quantiles(const series_model * m);   // the levels, in the file's order

// The input patch embedding: [time_encoding, values, mask] per patch -> d_model.
//   patches: n_rows * n_patches * (patch_size * 3), row-major
bool series_patch_embed(series_model * m, const float * patches, int n_rows, int n_patches,
                        std::vector<float> & out);

// The encoder stack over an already-embedded input.
//   embeds    n_rows * n_pos * d_model      one row per variate and covariate
//   group_ids n_rows                        rows of the same group attend to each other
//   mask      n_rows * n_pos                1 keeps a position, 0 hides it
//   stages    if non-null, receives "block_00".."block_NN" and "final_norm" for stage-by-stage checking
bool series_encode(series_model * m, const float * embeds, const float * group_ids, const float * mask,
                   int n_rows, int n_pos, std::vector<float> & out,
                   std::map<std::string, std::vector<float>> * stages = nullptr);

// Sense-T's own projector: encoder states -> soft tokens in the language model's embedding space.
// Adds the role (context / register / future) and variate embeddings, projects, and puts the result on
// the sphere the model's real token embeddings live on.
//   states  n_var * n_pos * d_model, the target variates' rows of the encoder output
//   roles   n_pos, one of 0 context, 1 register, 2 future
//   out     (n_var * n_pos) * hidden
bool series_project(series_model * m, const float * states, const int32_t * roles,
                    int n_var, int n_pos, std::vector<float> & out);

// What Chronos-2 builds before its encoder sees anything: standardise each row over its observed values,
// arcsinh it, cut it into patches, and hand each patch [time encoding, values, mask].
struct series_batch {
    std::vector<float> patched_ctx;    // n_rows * n_ctx * (patch_size * 3)
    std::vector<float> patched_fut;    // n_rows * n_fut * (patch_size * 3)
    std::vector<float> attn;           // n_rows * (n_ctx + 1 + n_fut), the register token included
    std::vector<float> loc, scale;     // per row: what it takes to read a forecast back in real units
    int n_rows = 0, n_ctx = 0, n_fut = 0;
};

// rows: n_rows sequences of n_steps values, NaN where nothing was observed.
// futures: n_rows sequences of n_fut_steps (NaN = unknown), or null. Give the caller's own padding:
// a covariate known to be zero ahead is zero with a live mask, which is not the same as unknown.
bool series_prepare(const series_hparams & h, const float * rows, int n_rows, int n_steps,
                    const float * futures, int n_fut_steps, int n_fut_patches, series_batch & out);

// Chronos' own quantile forecast, from the encoder's last n_fut states. Sense-T's residual head corrects
// this, so a runtime needs it even when the language model has the last word.
//   states  n_rows * n_pos * d_model (the encoder output)
//   loc, scale  per row, from series_prepare; pass null to leave the answer standardised
//   out     n_rows * n_quantiles * (n_fut * patch_size), quantile-major
bool series_forecast_base(series_model * m, const float * states, int n_rows, int n_pos, int n_fut,
                          const float * loc, const float * scale, std::vector<float> & out);

// Raw numbers to soft tokens: prepare, embed the patches, splice in the register token, run the encoder,
// keep the first n_target rows and project them. `roles` comes back alongside, one per token.
bool series_soft_tokens(series_model * m, const float * rows, int n_rows, int n_steps,
                        const float * futures, int n_fut_steps, int n_target, int n_fut_patches,
                        std::vector<float> & out, std::vector<int32_t> & roles);
