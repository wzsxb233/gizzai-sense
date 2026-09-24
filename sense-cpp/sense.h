// Gizzai Sense in C++: one forward pass, a calibrated probability per option, nothing generated.
//
// This is the whole of Sense on a device. llama.cpp gives the logits; the readout is the rest of this
// file. It is the same procedure as the released Python: render the prompt with the model's own chat
// template, resolve each option to the single token the model would emit for it, run one decode, and
// softmax over exactly those tokens.
//
//   sense::Model m("Gizzai-Sense-E2B-Q8_0.gguf");
//   auto a = m.noul("部署连续失败两次，客户已经看到 500 错误。", "需要立即处理吗？");
//   printf("%.3f\n", a.probs[0]);          // P(yes)
//
// Copyright (c) 2026 邓颐村 (Deng Yicun). Licensed under the Gizzai Sense License.
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

struct llama_model;
struct llama_context;
struct common_chat_templates;

namespace sense {

// what the caller gets back: the option keys and a probability for each, in the order asked
struct Answer {
    std::string              type;        // "noul" | "choice" | "score"
    std::vector<std::string> keys;
    std::vector<float>       probs;
    float                    confidence = 0.0f;
    std::string              best;        // the key with the highest probability
    float                    score = 0.0f;// score questions: the expected level
    int64_t                  us    = 0;   // microseconds for the forward pass
};

struct Options {
    int  n_ctx     = 4096;
    int  n_threads = 0;      // 0 = hardware concurrency
    int  n_gpu_layers = 0;   // Metal / Vulkan: set to 999 to offload everything
    bool verbose   = false;
    std::map<std::string, float> temperature;  // per question type; 1.0 unless the release says otherwise
};

class Model {
public:
    explicit Model(const std::string & gguf_path, const Options & opts = {});
    ~Model();

    Answer noul(const std::string & state, const std::string & question);
    Answer choice(const std::string & state, const std::string & question,
                  const std::vector<std::pair<std::string, std::string>> & options, bool open_set = false);
    Answer score(const std::string & state, const std::string & question,
                 const std::vector<std::string> & levels);

    // The same question when the state carries a series slot. The text before the slot, the series'
    // soft tokens and the text after are decoded as one sequence. Gemma 4 takes the per-layer input for
    // an embedding batch from the PAD row, which is exactly where Sense-T places these tokens, so this
    // needs no change to the engine — measured 0.0017 from transformers, inside quantisation.
    Answer noul_series(const std::string & state_with_mark, const std::string & question,
                       const float * soft, int n_soft, const std::string & mark = "[[SERIES]]");

    int n_embd() const;          // the width the soft tokens have to be

    // the prompt this would send, for debugging a mismatch against the reference implementation
    std::string debug_prompt(const std::string & state, const std::string & question_block) const;

private:
    struct Plan {
        std::string              type, block, primer;
        std::vector<std::string> labels, keys;
        std::vector<int32_t>     token_ids;
    };
    Plan    make_plan(const std::string & state, const std::string & type, const std::string & block,
                      const std::vector<std::string> & labels, const std::vector<std::string> & keys) const;
    Answer  run(const std::string & state, const Plan & plan);
    Answer  run_series(const std::string & state, const Plan & plan, const float * soft, int n_soft,
                       const std::string & mark);
    Answer  readout(const Plan & plan, const float * logits, int64_t us) const;
    std::string render(const std::string & state, const std::string & block) const;
    std::vector<int32_t> tokenize(const std::string & text, bool add_special = true) const;

    llama_model *          model_ = nullptr;
    llama_context *        ctx_   = nullptr;
    common_chat_templates * tmpls_ = nullptr;
    Options                opts_;
};

}  // namespace sense
