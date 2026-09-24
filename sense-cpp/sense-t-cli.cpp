// Sense-T end to end in C++: a series and a question in, a calibrated probability out.
//
//   ./build/sense-t-cli --model sense-t-E2B.gguf --series-model sense-t-series-F16.gguf \
//       --config <release>/sense_t.json --series series.json \
//       --state "这家店最近没有促销。" --question "未来 7 天里会不会有一天低于 950 件？"
//
// Four pieces, each checked against the released Python on its own:
//   prompt.cpp   the text half of the prompt, byte-identical on 200 cases
//   series.cpp   Chronos-2's encoder and Sense-T's projector, exact to 1.7e-06 at f32
//   sense.cpp    the three-chunk decode and the restricted-softmax readout
// This is the assembly, and the only thing it adds is wiring.
#include "prompt.h"
#include "sense.h"
#include "series.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

using json = nlohmann::json;

static json read_json(const std::string & path) {
    std::ifstream f(path);
    if (!f) {
        fprintf(stderr, "[sense-t] cannot open %s\n", path.c_str());
        exit(1);
    }
    json j;
    f >> j;
    return j;
}

int main(int argc, char ** argv) {
    std::string model, series_gguf, config, series_path, state, question;   // series_model is a type here
    int n_threads = 8, n_gpu_layers = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? argv[++i] : ""; };
        if      (a == "--model")         model        = next();
        else if (a == "--series-model")  series_gguf  = next();
        else if (a == "--config")        config       = next();
        else if (a == "--series")        series_path  = next();
        else if (a == "--state")         state        = next();
        else if (a == "--question")      question     = next();
        else if (a == "--threads")       n_threads    = atoi(next());
        else if (a == "--gpu-layers")    n_gpu_layers = atoi(next());
    }
    if (model.empty() || series_gguf.empty() || config.empty() || series_path.empty()) {
        fprintf(stderr, "usage: sense-t-cli --model X.gguf --series-model Y.gguf --config sense_t.json "
                        "--series series.json [--state S] --question Q\n");
        return 1;
    }

    const json cfg = read_json(config);
    const json sj  = read_json(series_path);
    const std::string mode    = cfg.value("mode", "mixed");
    const std::string summary = cfg.value("summary", "off");
    const int window       = cfg.value("window", 256);
    const int stats_window = cfg.value("stats_window", 0);

    series_text s;
    for (const auto & n : sj["names"]) s.names.push_back(n.get<std::string>());
    for (const auto & r : sj["values"]) s.values.push_back(r.get<std::vector<double>>());
    s.unit = sj.value("unit", "");
    s.step = sj.value("step", "");

    // 1. the text half of the prompt, with the slot the series will occupy
    const std::string st = state_text(s, state, "", mode, window, stats_window, summary);

    // 2. the series half: the encoder reads only the most recent `window` steps, as the release does
    series_model * sm = series_load(series_gguf.c_str(), n_threads);
    if (!sm) return 1;
    const series_hparams & h = series_hp(sm);
    const int n_fut_patches = 2;
    const int n_fut_steps   = n_fut_patches * h.patch_size;
    const int n_target      = (int) s.values.size();

    int n_steps = 0;
    for (const auto & row : s.values) n_steps = (int) row.size() > n_steps ? (int) row.size() : n_steps;
    n_steps = n_steps > window ? window : n_steps;

    std::vector<float> rows((size_t) n_target * n_steps);
    for (int r = 0; r < n_target; ++r) {
        const std::vector<double> & v = s.values[r];
        for (int t = 0; t < n_steps; ++t) {
            const int src = (int) v.size() - n_steps + t;       // the last n_steps of the row
            rows[(size_t) r * n_steps + t] = src >= 0 ? (float) v[src] : NAN;
        }
    }
    std::vector<float> futures((size_t) n_target * n_fut_steps, NAN);   // the target's future is unknown

    std::vector<float>   soft;
    std::vector<int32_t> roles;
    if (!series_soft_tokens(sm, rows.data(), n_target, n_steps, futures.data(), n_fut_steps,
                            n_target, n_fut_patches, soft, roles)) {
        fprintf(stderr, "[sense-t] the series encoder failed\n");
        return 1;
    }
    const int n_soft = (int) (soft.size() / h.hidden);

    // 3. the language model: prompt text, soft tokens, prompt text, then read the options
    sense::Options opts;
    opts.n_ctx = 4096;
    opts.n_threads = n_threads;
    opts.n_gpu_layers = n_gpu_layers;
    if (cfg.contains("temperature")) {
        const json & temps = cfg["temperature"];                 // bound first: iterating a temporary
        for (auto it = temps.begin(); it != temps.end(); ++it) { // once applied some and skipped others
            opts.temperature[it.key()] = it.value().get<float>();
        }
    }
    sense::Model m(model, opts);
    if ((int) soft.size() / n_soft != m.n_embd()) {
        fprintf(stderr, "[sense-t] soft tokens are %d wide, the model wants %d\n",
                (int) soft.size() / n_soft, m.n_embd());
        return 1;
    }

    printf("[sense-t] %d series x %d steps -> %d soft tokens of %d; mode %s, window %d\n",
           n_target, n_steps, n_soft, m.n_embd(), mode.c_str(), window);
    const sense::Answer a = m.noul_series(st, question, soft.data(), n_soft);
    printf("  question : %s\n", question.c_str());
    printf("  P(yes)   : %.4f      confidence %.3f      %.0f ms\n", a.probs[0], a.confidence, a.us / 1000.0);
    series_free(sm);
    return 0;
}
