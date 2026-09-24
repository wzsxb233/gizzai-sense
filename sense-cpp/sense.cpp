#include "sense.h"

#include "chat.h"
#include "common.h"
#include "llama.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <thread>

namespace sense {

namespace {

const char * SYSTEM_PROMPT =
    "You are a decision engine. You read a STATE and answer one question about "
    "it by emitting exactly one label. Never explain, never add words.";
const char * ANSWER_PRIMER = "Answer:";
// Gemma makes a space its own token, so ending the primer with one puts each label in the next token.
// Tried in order, exactly as the released format does; the first distinct tokenisation wins.
const std::pair<const char *, const char *> PRIMER_VARIANTS[] = {{" ", ""}, {"", " "}};
const char * OTHER             = "__other__";
const char * OTHER_DESCRIPTION = "None of the listed options fits";

std::string letters(size_t i) { return std::string(1, char('A' + i)); }

float entropy_confidence(const std::vector<float> & p) {
    if (p.size() < 2) return 1.0f;
    double h = 0.0;
    for (float x : p) h -= x * std::log(std::max(x, 1e-12f));
    return float(std::max(0.0, std::min(1.0, 1.0 - h / std::log(double(p.size())))));
}

}  // namespace

Model::Model(const std::string & gguf_path, const Options & opts) : opts_(opts) {
    llama_backend_init();
    auto mparams          = llama_model_default_params();
    mparams.n_gpu_layers  = opts.n_gpu_layers;
    model_ = llama_model_load_from_file(gguf_path.c_str(), mparams);
    if (!model_) throw std::runtime_error("could not load " + gguf_path);

    auto cparams    = llama_context_default_params();
    cparams.n_ctx   = opts.n_ctx;
    cparams.n_batch = opts.n_ctx;
    cparams.n_threads = opts.n_threads > 0 ? opts.n_threads : int(std::thread::hardware_concurrency());
    cparams.n_threads_batch = cparams.n_threads;
    ctx_ = llama_init_from_model(model_, cparams);
    if (!ctx_) throw std::runtime_error("could not create a context for " + gguf_path);

    auto tmpls = common_chat_templates_init(model_, "");
    tmpls_ = tmpls.release();                     // owned for the life of the model
    if (!tmpls_) throw std::runtime_error("the GGUF carries no chat template");

    // The released temperatures are part of the model, not of the runtime: without them a port answers
    // with different confidence than the model was calibrated to. Read them from sense_readout.json
    // beside the weights unless the caller passed their own.
    if (opts_.temperature.empty()) {
        const auto slash = gguf_path.find_last_of("/\\");
        const std::string dir = slash == std::string::npos ? std::string(".") : gguf_path.substr(0, slash);
        std::ifstream in(dir + "/sense_readout.json");
        if (in) {
            try {
                nlohmann::json j;
                in >> j;
                // bind the object first: iterating `.items()` of a temporary is undefined behaviour,
                // and it silently applied some temperatures and not others
                const nlohmann::json temps = j.value("temperature", nlohmann::json::object());
                for (auto it = temps.begin(); it != temps.end(); ++it)
                    opts_.temperature[it.key()] = it.value().get<float>();
                if (opts_.verbose)
                    fprintf(stderr, "[sense] readout temperatures: %zu entries\n", opts_.temperature.size());
            } catch (const std::exception & e) {
                fprintf(stderr, "[sense] sense_readout.json unreadable (%s); using temperature 1.0\n", e.what());
            }
        } else if (opts_.verbose) {
            fprintf(stderr, "[sense] no sense_readout.json beside the model; using temperature 1.0\n");
        }
    }
}

Model::~Model() {
    if (ctx_)   llama_free(ctx_);
    if (model_) llama_model_free(model_);
    llama_backend_free();
}

std::string Model::render(const std::string & state, const std::string & block) const {
    common_chat_templates_inputs in;
    in.use_jinja             = true;
    in.add_generation_prompt = true;
    in.enable_thinking       = false;
    common_chat_msg sys;
    sys.role = "system";
    sys.content = SYSTEM_PROMPT;
    common_chat_msg user;
    user.role = "user";
    user.content = "<state>\n" + state + "\n</state>\n\n" + block;
    in.messages = {sys, user};
    return common_chat_templates_apply(tmpls_, in).prompt;
}

std::string Model::debug_prompt(const std::string & state, const std::string & block) const {
    return render(state, block) + ANSWER_PRIMER;
}

std::vector<int32_t> Model::tokenize(const std::string & text, bool add_special) const {
    // Exactly one BOS, whoever puts it there. llama.cpp's templates leave BOS out and expect the
    // tokenizer to add it (common_chat_templates_apply even strips a duplicate), which is the opposite
    // of transformers, whose template emits it. Getting this wrong in either direction changes the
    // prompt: two BOS moved a choice probability by 0.17, none weakened every answer.
    // Only the first chunk of a sequence gets one: the series path tokenises the text after the slot
    // separately, and a second BOS in the middle of a prompt is the same bug seen from another angle.
    return common_tokenize(llama_model_get_vocab(model_), text, add_special, /*parse_special*/ true);
}

Model::Plan Model::make_plan(const std::string & state, const std::string & type, const std::string & block,
                             const std::vector<std::string> & labels, const std::vector<std::string> & keys) const {
    const std::string base = render(state, block) + ANSWER_PRIMER;
    for (const auto & variant : PRIMER_VARIANTS) {
        const std::string prefix = base + variant.first;
        const auto        a      = tokenize(prefix);
        std::vector<int32_t> ids;
        bool ok = true;
        for (const auto & lab : labels) {
            const auto b = tokenize(prefix + variant.second + lab);
            size_t i = 0;
            while (i < a.size() && i < b.size() && a[i] == b[i]) i++;
            if (i != a.size() || i >= b.size()) { ok = false; break; }   // the label must start a clean token
            ids.push_back(b[i]);
        }
        if (!ok) continue;
        std::vector<int32_t> uniq = ids;
        std::sort(uniq.begin(), uniq.end());
        if (std::unique(uniq.begin(), uniq.end()) != uniq.end()) continue;   // labels collide
        if (getenv("SENSE_DEBUG_PLAN")) {
            fprintf(stderr, "[plan] %s primer=%s%s ids=", type.c_str(), ANSWER_PRIMER,
                    variant.first[0] ? "' '" : "");
            for (int32_t id : ids) fprintf(stderr, "%d ", id);
            fprintf(stderr, "\n");
        }
        return Plan{type, block, std::string(ANSWER_PRIMER) + variant.first, labels, keys, ids};
    }
    throw std::runtime_error("no usable label tokenisation for this question");
}

Answer Model::run(const std::string & state, const Plan & plan) {
    const auto  t0     = std::chrono::steady_clock::now();
    const std::string text = render(state, plan.block) + plan.primer;
    if (getenv("SENSE_DEBUG_PROMPT")) fprintf(stderr, "[prompt]%s[/prompt]\n", text.c_str());
    const auto  tokens = tokenize(text);
    if (getenv("SENSE_DEBUG_TOKENS")) {
        fprintf(stderr, "[tokens] n=%zu first=%d %d %d last=%d %d %d\n", tokens.size(), tokens[0], tokens[1],
                tokens[2], tokens[tokens.size() - 3], tokens[tokens.size() - 2], tokens[tokens.size() - 1]);
    }
    if (int(tokens.size()) > opts_.n_ctx)
        throw std::runtime_error("prompt is longer than the context window");

    llama_memory_clear(llama_get_memory(ctx_), true);
    llama_batch batch = llama_batch_get_one(const_cast<int32_t *>(tokens.data()), int32_t(tokens.size()));
    if (llama_decode(ctx_, batch) != 0) throw std::runtime_error("llama_decode failed");

    const float * logits = llama_get_logits_ith(ctx_, -1);
    if (!logits) throw std::runtime_error("no logits returned");
    return readout(plan, logits,
                   std::chrono::duration_cast<std::chrono::microseconds>(
                       std::chrono::steady_clock::now() - t0).count());
}

// logits -> a calibrated probability per option: the whole of Sense, once the forward pass is done
Answer Model::readout(const Plan & plan, const float * logits, int64_t us) const {
    auto it = opts_.temperature.find(plan.type);
    const float temp = it != opts_.temperature.end() ? it->second : 1.0f;

    std::vector<float> sel;
    sel.reserve(plan.token_ids.size());
    for (int32_t id : plan.token_ids) sel.push_back(logits[id] / temp);
    const float m = *std::max_element(sel.begin(), sel.end());
    double sum = 0.0;
    for (float & x : sel) { x = std::exp(x - m); sum += x; }
    for (float & x : sel) x = float(x / sum);

    Answer out;
    out.type  = plan.type;
    out.keys  = plan.keys;
    out.probs = sel;
    size_t best = size_t(std::max_element(sel.begin(), sel.end()) - sel.begin());
    out.best  = plan.keys[best];
    out.confidence = plan.type == "noul" ? std::fabs(2.0f * sel[0] - 1.0f) : entropy_confidence(sel);
    if (plan.type == "score") {
        double s = 0.0;
        for (size_t i = 0; i < sel.size(); i++) s += double(i) * sel[i];
        out.score = float(s);
    }
    out.us = us;
    return out;
}

// one sequence, three decodes: the prompt up to the series slot, the series itself as embeddings, and
// the rest of the prompt. llama.cpp substitutes the PAD row's per-layer input for an embedding batch,
// which is what Sense-T's slots hold, so the two halves of the model agree without any engine change.
static bool decode_tokens(llama_context * ctx, const std::vector<int32_t> & toks, int pos0, bool last_logits) {
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

int Model::n_embd() const { return llama_model_n_embd_inp(model_); }

Answer Model::run_series(const std::string & state, const Plan & plan, const float * soft, int n_soft,
                         const std::string & mark) {
    const auto        t0   = std::chrono::steady_clock::now();
    const std::string text = render(state, plan.block) + plan.primer;
    if (getenv("SENSE_DEBUG_PROMPT")) fprintf(stderr, "[prompt]%s[/prompt]\n", text.c_str());
    const size_t      at   = text.find(mark);
    if (at == std::string::npos) throw std::runtime_error("the state carries no series slot");
    const auto left  = tokenize(text.substr(0, at), /*add_special*/ true);
    const auto right = tokenize(text.substr(at + mark.size()), /*add_special*/ false);
    if (int(left.size()) + n_soft + int(right.size()) > opts_.n_ctx)
        throw std::runtime_error("prompt is longer than the context window");
    if (getenv("SENSE_DEBUG_TOKENS")) {
        fprintf(stderr, "[series] %zu text + %d soft + %zu text tokens\n", left.size(), n_soft, right.size());
    }

    llama_memory_clear(llama_get_memory(ctx_), true);
    if (!decode_tokens(ctx_, left, 0, false)
        || !decode_embd(ctx_, soft, n_soft, n_embd(), (int) left.size())
        || !decode_tokens(ctx_, right, (int) left.size() + n_soft, true))
        throw std::runtime_error("llama_decode failed");

    const float * logits = llama_get_logits_ith(ctx_, -1);
    if (!logits) throw std::runtime_error("no logits returned");
    return readout(plan, logits,
                   std::chrono::duration_cast<std::chrono::microseconds>(
                       std::chrono::steady_clock::now() - t0).count());
}

Answer Model::noul_series(const std::string & state_with_mark, const std::string & question,
                          const float * soft, int n_soft, const std::string & mark) {
    const std::string block = "Question: " + question + "\nReply with yes or no.\n";
    return run_series(state_with_mark, make_plan(state_with_mark, "noul", block, {"yes", "no"}, {"yes", "no"}),
                      soft, n_soft, mark);
}

Answer Model::noul(const std::string & state, const std::string & question) {
    const std::string block = "Question: " + question + "\nReply with yes or no.\n";
    return run(state, make_plan(state, "noul", block, {"yes", "no"}, {"yes", "no"}));
}

Answer Model::choice(const std::string & state, const std::string & question,
                     const std::vector<std::pair<std::string, std::string>> & options, bool open_set) {
    auto opts = options;
    if (open_set) {
        bool present = false;
        for (const auto & kv : opts) present = present || kv.first == OTHER;
        if (!present) opts.emplace_back(OTHER, OTHER_DESCRIPTION);
    }
    if (opts.size() < 2 || opts.size() > 26) throw std::runtime_error("choice needs 2-26 options");
    std::string lines;
    std::vector<std::string> labels, keys;
    for (size_t i = 0; i < opts.size(); i++) {
        labels.push_back(letters(i));
        keys.push_back(opts[i].first);
        lines += letters(i) + ") " + opts[i].second + (i + 1 < opts.size() ? "\n" : "");
    }
    const std::string block = "Question: " + question + "\nOptions:\n" + lines +
                              "\nReply with the single letter of the best option.\n";
    return run(state, make_plan(state, "choice", block, labels, keys));
}

Answer Model::score(const std::string & state, const std::string & question,
                    const std::vector<std::string> & levels) {
    if (levels.size() < 2 || levels.size() > 10) throw std::runtime_error("score needs 2-10 levels");
    std::string lines;
    std::vector<std::string> labels, keys;
    for (size_t i = 0; i < levels.size(); i++) {
        labels.push_back(std::to_string(i));
        keys.push_back(levels[i]);
        lines += std::to_string(i) + " = " + levels[i] + (i + 1 < levels.size() ? "\n" : "");
    }
    const std::string block = "Question: " + question + "\nLevels:\n" + lines +
                              "\nReply with the single digit of the level that fits best.\n";
    return run(state, make_plan(state, "score", block, labels, keys));
}

}  // namespace sense
