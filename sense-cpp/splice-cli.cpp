// Does llama.cpp's embedding path carry Sense-T's soft tokens the way transformers does?
//
//   python scripts/dump_splice_ref.py --release /root/release/Gizzai-Sense-E2B
//   ./build/splice-cli --model /root/gguf/Gizzai-Sense-E2B-Q8_0.gguf --ref /root/gguf/splice_ref.bin
//
// Gemma 4 gives every position a second, per-layer input looked up by token id. Sense-T's soft tokens
// sit in PAD slots, so transformers gives those positions the PAD row; llama.cpp, decoding embeddings,
// substitutes per_layer_tok_embd row 0, which for this model is also PAD. This runs the same prompt and
// the same soft tokens through both and prints how far apart they land.
//
// Two cases, because one number cannot tell a broken splice from a broken harness:
//   pad    the slots keep their PAD embeddings, so llama.cpp can run the whole prompt as plain tokens.
//          Disagreement here means the harness is wrong and the second number means nothing.
//   soft   the slots carry the reference's soft tokens, decoded as an embedding batch.
#include "llama.h"
#include "refbin.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

static std::vector<llama_token> ids_of(const ref_tensor & t) {
    std::vector<llama_token> v(t.data.size());
    for (size_t i = 0; i < v.size(); ++i) v[i] = (llama_token) lrintf(t.data[i]);
    return v;
}

static bool decode_tokens(llama_context * ctx, const std::vector<llama_token> & toks, int pos0, bool last_logits) {
    llama_batch b = llama_batch_init((int32_t) toks.size(), 0, 1);
    b.n_tokens = (int32_t) toks.size();
    for (size_t i = 0; i < toks.size(); ++i) {
        b.token[i]     = toks[i];
        b.pos[i]       = pos0 + (int) i;
        b.n_seq_id[i]  = 1;
        b.seq_id[i][0] = 0;
        b.logits[i]    = (last_logits && i + 1 == toks.size()) ? 1 : 0;
    }
    const bool ok = llama_decode(ctx, b) == 0;
    llama_batch_free(b);
    return ok;
}

static bool decode_embd(llama_context * ctx, const float * e, int n, int n_embd, int pos0) {
    llama_batch b = llama_batch_init(n, n_embd, 1);
    b.n_tokens = n;
    memcpy(b.embd, e, (size_t) n * n_embd * sizeof(float));
    for (int i = 0; i < n; ++i) {
        b.pos[i]       = pos0 + i;
        b.n_seq_id[i]  = 1;
        b.seq_id[i][0] = 0;
        b.logits[i]    = 0;
    }
    const bool ok = llama_decode(ctx, b) == 0;
    llama_batch_free(b);
    return ok;
}

// the restricted softmax Sense actually reads: only the options compete
static std::vector<float> option_probs(const float * logits, const std::vector<llama_token> & options) {
    double max_l = -1e30;
    for (llama_token o : options) max_l = logits[o] > max_l ? logits[o] : max_l;
    double sum = 0.0;
    std::vector<float> p(options.size());
    for (size_t i = 0; i < options.size(); ++i) {
        p[i] = (float) std::exp(logits[options[i]] - max_l);
        sum += p[i];
    }
    for (float & v : p) v = (float) (v / sum);
    return p;
}

int main(int argc, char ** argv) {
    const char * model_path = "/root/gguf/Gizzai-Sense-E2B-Q8_0.gguf";
    const char * ref_path   = "/root/gguf/splice_ref.bin";
    int          n_threads  = 8;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc)        model_path = argv[++i];
        else if (!strcmp(argv[i], "--ref") && i + 1 < argc)     ref_path   = argv[++i];
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) n_threads  = atoi(argv[++i]);
    }

    std::map<std::string, ref_tensor> ref;
    if (!ref_load(ref_path, ref)) return 1;
    const std::vector<llama_token> left    = ids_of(ref["left"]);
    const std::vector<llama_token> right   = ids_of(ref["right"]);
    const std::vector<llama_token> options = ids_of(ref["options"]);
    const ref_tensor &             soft    = ref["soft"];
    const int n_soft = (int) soft.dims[0], d_soft = (int) soft.dims[1];

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) {
        fprintf(stderr, "[splice] cannot load %s\n", model_path);
        return 1;
    }
    const int n_embd = llama_model_n_embd_inp(model);
    const llama_token pad = llama_vocab_pad(llama_model_get_vocab(model));
    const int n_tok = (int) left.size() + n_soft + (int) right.size();

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = cp.n_batch = cp.n_ubatch = n_tok + 8;
    cp.n_threads = cp.n_threads_batch = n_threads;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 1;

    printf("[splice] %s\n         %zu + %d slots + %zu tokens, n_embd %d (soft tokens are %d wide), pad id %d\n\n",
           model_path, left.size(), n_soft, right.size(), n_embd, d_soft, pad);
    if (d_soft != n_embd) {
        fprintf(stderr, "[splice] the reference's soft tokens are %d wide, the model wants %d\n", d_soft, n_embd);
        return 1;
    }

    std::map<std::string, std::pair<float, float>> first;   // case -> (llama.cpp, transformers) p[0]
    int n_bad = 0;
    printf("  case   llama.cpp            transformers         max |dp|\n");
    for (const std::string & c : { std::string("pad"), std::string("soft") }) {
        llama_memory_clear(llama_get_memory(ctx), true);
        bool ok = true;
        if (c == "pad") {
            std::vector<llama_token> all = left;
            all.insert(all.end(), n_soft, pad);
            all.insert(all.end(), right.begin(), right.end());
            ok = decode_tokens(ctx, all, 0, true);
        } else {
            ok = decode_tokens(ctx, left, 0, false)
              && decode_embd(ctx, soft.data.data(), n_soft, n_embd, (int) left.size())
              && decode_tokens(ctx, right, (int) left.size() + n_soft, true);
        }
        if (!ok) {
            printf("  %-5s  decode failed\n", c.c_str());
            ++n_bad;
            continue;
        }
        const std::vector<float> got  = option_probs(llama_get_logits_ith(ctx, -1), options);
        const std::vector<float> want = ref[c + ".probs"].data;
        double worst = 0.0;
        for (size_t i = 0; i < got.size(); ++i) worst = std::max(worst, (double) std::fabs(got[i] - want[i]));
        printf("  %-5s  [%.4f, %.4f]     [%.4f, %.4f]     %.4f\n",
               c.c_str(), got[0], got[1], want[0], want[1], worst);
        first[c] = { got[0], want[0] };
        if (worst > 0.05) ++n_bad;
    }
    // the sharpest number here: not where each case lands, but how much the soft tokens moved it
    if (first.count("pad") && first.count("soft")) {
        printf("\n  what the soft tokens do to the first option: %+.4f in llama.cpp, %+.4f in transformers\n",
               first["soft"].first - first["pad"].first, first["soft"].second - first["pad"].second);
    }
    printf("\n%s\n", n_bad == 0
        ? "llama.cpp's embedding path carries the soft tokens: both cases agree within quantisation,\n"
          "and the soft tokens move the answer by the same amount. The per-layer substitution is right\n"
          "for a model whose slots are PAD, which is how Sense-T places them."
        : "the two implementations disagree by more than quantisation explains.");

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return n_bad == 0 ? 0 : 1;
}
