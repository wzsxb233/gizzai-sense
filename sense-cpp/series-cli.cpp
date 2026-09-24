// Check the ggml port of Chronos-2 against the real encoder, one stage at a time.
//
//   python scripts/dump_chronos_ref.py --out /root/gguf/chronos_ref.npz
//   python scripts/ref_to_bin.py
//   ./build/series-cli --model /root/gguf/sense-t-series-F16.gguf --ref /root/gguf/chronos_ref.bin
//
// A single end-to-end number tells you the port is wrong without telling you where. This prints the
// error after the patch embedding, after every one of the twelve blocks, and after the final norm, so
// the first row that goes red is the bug.
#include "series.h"
#include "refbin.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// As a fraction of the stage's own spread. An F32 file lands at 1e-5 of it; F16 weights cost about 1%
// on the worst single element and a thousandth of that on average, so 2% passes F16 and still catches
// anything structural — a wrong mask or a missing rotation is off by the size of the signal, not 1%.
static int   n_fail = 0;
static float tol    = 0.02f;

static void check(const char * stage, const std::vector<float> & got, const ref_tensor & ref) {
    if (got.size() != ref.data.size()) {
        printf("  %-14s  SHAPE  got %zu values, reference has %zu\n", stage, got.size(), ref.data.size());
        ++n_fail;
        return;
    }
    double max_abs = 0.0, sum_abs = 0.0, sum = 0.0, sum2 = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double d = std::fabs((double) got[i] - (double) ref.data[i]);
        max_abs = d > max_abs ? d : max_abs;
        sum_abs += d;
        sum  += ref.data[i];
        sum2 += (double) ref.data[i] * ref.data[i];
    }
    const double n    = (double) got.size();
    const double sd   = std::sqrt(sum2 / n - (sum / n) * (sum / n));
    const double rel  = sd > 0 ? max_abs / sd : max_abs;
    const bool   ok   = rel <= tol;
    if (!ok) ++n_fail;
    printf("  %-14s  max %.2e   mean %.2e   ref sd %.4f   max/sd %5.2f%%   mean/sd %.4f%%  %s\n",
           stage, max_abs, sum_abs / n, sd, 100.0 * rel,
           sd > 0 ? 100.0 * (sum_abs / n) / sd : 0.0, ok ? "ok" : "**");
}

int main(int argc, char ** argv) {
    const char * model_path = "/root/gguf/sense-t-series-F16.gguf";
    const char * ref_path   = "/root/gguf/chronos_ref.bin";
    int          n_threads  = 4;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc)        model_path = argv[++i];
        else if (!strcmp(argv[i], "--ref") && i + 1 < argc)     ref_path   = argv[++i];
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) n_threads  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tol") && i + 1 < argc)     tol        = atof(argv[++i]);
    }

    std::map<std::string, ref_tensor> ref;
    if (!ref_load(ref_path, ref)) return 1;

    series_model * m = series_load(model_path, n_threads);
    if (!m) return 1;
    const series_hparams & h = series_hp(m);
    printf("[series] %s\n         d_model %d, %d blocks, %d heads x %d, d_ff %d, patch %d, %d quantiles\n",
           model_path, h.d_model, h.n_layers, h.n_heads, h.d_kv, h.d_ff, h.patch_size, h.n_quantiles);
    printf("         reference: %s (%zu tensors), tolerance %.2g of each stage's spread\n\n",
           ref_path, ref.size(), tol);

    // 1. the input patch embedding, once per call site: context patches, then future patches
    printf("patch embedding\n");
    static const char * const PAIRS[][2] = { { "patch_in.x", "patch_embed" },
                                             { "patch_in.x_1", "patch_embed_1" } };
    for (const auto & pair : PAIRS) {
        auto in = ref.find(pair[0]), want = ref.find(pair[1]);
        if (in == ref.end() || want == ref.end()) continue;
        std::vector<float> got;
        series_patch_embed(m, in->second.data.data(), (int) in->second.dims[0], (int) in->second.dims[1], got);
        check(pair[1], got, want->second);
    }

    // 2. the encoder stack, block by block
    auto embeds = ref.find("encoder.in.inputs_embeds");
    auto gids   = ref.find("encoder.in.group_ids");
    auto amask  = ref.find("encoder.in.attention_mask");
    if (embeds == ref.end() || gids == ref.end() || amask == ref.end()) {
        fprintf(stderr, "[series-cli] the reference dump has no encoder inputs\n");
        series_free(m);
        return 1;
    }
    const int n_rows = (int) embeds->second.dims[0];
    const int n_pos  = (int) embeds->second.dims[1];
    printf("\nencoder: %d rows x %d positions\n", n_rows, n_pos);

    std::vector<float>                          out;
    std::map<std::string, std::vector<float>>   stages;
    series_encode(m, embeds->second.data.data(), gids->second.data.data(), amask->second.data.data(),
                  n_rows, n_pos, out, &stages);
    for (auto & kv : stages) {
        auto want = ref.find(kv.first);
        if (want != ref.end()) check(kv.first.c_str(), kv.second, want->second);
    }

    // 3. Sense-T's projector: the states the model keeps, turned into soft tokens
    auto st = ref.find("states"), roles = ref.find("roles"), proj = ref.find("projected");
    if (st != ref.end() && roles != ref.end() && proj != ref.end()) {
        printf("\nprojector: %u variates x %u positions -> %d\n", st->second.dims[0], st->second.dims[1], h.hidden);
        std::vector<int32_t> rid(roles->second.data.size());
        for (size_t i = 0; i < rid.size(); ++i) rid[i] = (int32_t) lrintf(roles->second.data[i]);
        std::vector<float> got;
        series_project(m, st->second.data.data(), rid.data(), (int) st->second.dims[0], (int) st->second.dims[1], got);
        check("projected", got, proj->second);
    }

    // 3b. Chronos' own quantile forecast, still standardised, straight from the encoder output
    auto fin = ref.find("final_norm"), opatch = ref.find("out_patch");
    if (fin != ref.end() && opatch != ref.end()) {
        const int rows = (int) fin->second.dims[0], pos = (int) fin->second.dims[1];
        const int n_fut = (int) opatch->second.dims[1];
        std::vector<float> got;
        series_forecast_base(m, fin->second.data.data(), rows, pos, n_fut, nullptr, nullptr, got);
        // the reference is patch-major, this is quantile-major: compare it the way the reference stores it
        std::vector<float> flat(got.size());
        const int Q = h.n_quantiles, P = h.output_patch_size;
        for (int r = 0; r < rows; ++r)
            for (int n = 0; n < n_fut; ++n)
                for (int q = 0; q < Q; ++q)
                    for (int p = 0; p < P; ++p)
                        flat[((size_t) r * n_fut + n) * Q * P + (size_t) q * P + p] =
                            got[((size_t) r * Q + q) * n_fut * P + (size_t) n * P + p];
        printf("\nquantile head: %d rows x %d future patches -> %d quantiles x %d steps\n", rows, n_fut, Q, P);
        check("out_patch", flat, opatch->second);

        // and in real units, which exercises the other half: sinh, then the row's own loc and scale
        auto ls = ref.find("loc_scale");
        if (ls != ref.end()) {
            std::vector<float> loc(rows), sc(rows), real;
            for (int r = 0; r < rows; ++r) { loc[r] = ls->second.data[2 * r]; sc[r] = ls->second.data[2 * r + 1]; }
            series_forecast_base(m, fin->second.data.data(), rows, pos, n_fut, loc.data(), sc.data(), real);
            const std::vector<float> & lv = series_quantiles(m);
            int i10 = 0, i50 = 0, i90 = 0;
            for (int q = 0; q < Q; ++q) {
                if (std::fabs(lv[q] - 0.1f) < std::fabs(lv[i10] - 0.1f)) i10 = q;
                if (std::fabs(lv[q] - 0.5f) < std::fabs(lv[i50] - 0.5f)) i50 = q;
                if (std::fabs(lv[q] - 0.9f) < std::fabs(lv[i90] - 0.9f)) i90 = q;
            }
            printf("  the series averages %.1f; the next 6 steps, target variate:\n    ",
                   ls->second.data[0]);
            for (int s = 0; s < 6 && s < n_fut * P; ++s) {
                printf("%.1f/%.1f/%.1f  ", real[(size_t) i10 * n_fut * P + s],
                       real[(size_t) i50 * n_fut * P + s], real[(size_t) i90 * n_fut * P + s]);
            }
            printf("\n    (q%.2f / q%.2f / q%.2f)\n", lv[i10], lv[i50], lv[i90]);
            bool ordered = true;
            for (int s = 0; s < n_fut * P; ++s) {
                ordered &= real[(size_t) i10 * n_fut * P + s] <= real[(size_t) i50 * n_fut * P + s]
                        && real[(size_t) i50 * n_fut * P + s] <= real[(size_t) i90 * n_fut * P + s];
            }
            printf("  %-14s  %s\n", "quantile order", ordered ? "low <= median <= high at every step"
                                                              : "** crossed, which cannot be right");
            if (!ordered) ++n_fail;
        }
    }

    // 4. the whole way: raw numbers and a covariate in, soft tokens out, nothing borrowed from torch
    auto vals = ref.find("values"), pcov = ref.find("past_cov"), fcov = ref.find("future_cov");
    if (vals != ref.end() && pcov != ref.end() && fcov != ref.end() && proj != ref.end()) {
        const int n_target = (int) vals->second.dims[0];
        const int n_steps  = (int) vals->second.dims[1];
        const int n_rows   = n_target + 1;                 // the series, then its covariate
        const int n_fut    = 2 * h.patch_size;
        printf("\nend to end: %d values + %d covariate steps -> soft tokens\n", n_steps, n_steps);

        std::vector<float> rows((size_t) n_rows * n_steps);
        memcpy(rows.data(), vals->second.data.data(), (size_t) n_target * n_steps * sizeof(float));
        memcpy(rows.data() + (size_t) n_target * n_steps, pcov->second.data.data(), n_steps * sizeof(float));

        // the target's own future is unknown; the covariate's is known where given and zero after,
        // which is what Sense-T feeds and is not the same as unobserved
        std::vector<float> futures((size_t) n_rows * n_fut, 0.0f);
        for (int t = 0; t < n_fut; ++t) futures[t] = NAN;
        for (size_t i = 0; i < fcov->second.data.size() && (int) i < n_fut; ++i) {
            futures[(size_t) n_target * n_fut + i] = fcov->second.data[i];
        }

        std::vector<float>   got;
        std::vector<int32_t> got_roles;
        series_soft_tokens(m, rows.data(), n_rows, n_steps, futures.data(), n_fut, n_target, 2, got, got_roles);
        bool roles_ok = got_roles.size() == roles->second.data.size();
        for (size_t i = 0; roles_ok && i < got_roles.size(); ++i) {
            roles_ok = got_roles[i] == (int32_t) lrintf(roles->second.data[i]);
        }
        printf("  %-14s  %s\n", "roles", roles_ok ? "match" : "** differ from the reference");
        if (!roles_ok) ++n_fail;
        check("soft tokens", got, proj->second);
    }

    printf("\n%s\n", n_fail == 0 ? "every stage matches the reference encoder."
                                 : "stages marked ** differ from the reference; the first one is the bug.");
    series_free(m);
    return n_fail == 0 ? 0 : 1;
}
