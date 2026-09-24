// A tiny command line over sense.h: the same three question types the SDK exposes on a phone.
//
//   sense-cli -m Gizzai-Sense-E2B-Q8_0.gguf --demo
//   sense-cli -m model.gguf --state "部署连续失败两次。" --noul "需要立即处理吗？"
//   sense-cli -m model.gguf --state "..." --choice "Which team owns this?" infra=Infrastructure billing=Billing
#include "sense.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static void print(const sense::Answer & a) {
    printf("{\"type\": \"%s\", \"best\": \"%s\", \"confidence\": %.4f, \"ms\": %.1f, \"probabilities\": {",
           a.type.c_str(), a.best.c_str(), a.confidence, a.us / 1000.0);
    for (size_t i = 0; i < a.keys.size(); i++)
        printf("%s\"%s\": %.6f", i ? ", " : "", a.keys[i].c_str(), a.probs[i]);
    printf("}%s}\n", a.type == "score" ? (", \"score\": " + std::to_string(a.score)).c_str() : "");
}

int main(int argc, char ** argv) {
    std::string model, state, noul_q, choice_q, score_q;
    std::vector<std::pair<std::string, std::string>> options;
    std::vector<std::string> levels;
    bool demo = false, open_set = false, dump_prompt = false;
    std::string jsonl;
    sense::Options opts;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "-m" || a == "--model") model = next();
        else if (a == "--state")   state   = next();
        else if (a == "--noul")    noul_q  = next();
        else if (a == "--score")   score_q = next();
        else if (a == "--choice")  choice_q = next();
        else if (a == "--open-set") open_set = true;
        else if (a == "--demo")     demo = true;
        else if (a == "--dump-prompt") { dump_prompt = true; }
        else if (a == "--jsonl") jsonl = next();
        else if (a == "--threads")  opts.n_threads = atoi(next());
        else if (a == "--gpu-layers") opts.n_gpu_layers = atoi(next());
        else if (a == "--level")    levels.push_back(next());
        else if (a.find('=') != std::string::npos)
            options.emplace_back(a.substr(0, a.find('=')), a.substr(a.find('=') + 1));
        else { fprintf(stderr, "unknown argument: %s\n", a.c_str()); return 2; }
    }
    if (model.empty()) { fprintf(stderr, "usage: sense-cli -m model.gguf [--demo | --state S --noul Q | ...]\n"); return 2; }

    try {
        sense::Model m(model, opts);
        if (dump_prompt) {   // byte-for-byte comparison against the reference implementation's prompt
            const std::string block = noul_q.empty() ? "Question: 需要立即处理吗？\nReply with yes or no.\n"
                                                     : "Question: " + noul_q + "\nReply with yes or no.\n";
            const std::string p = m.debug_prompt(state, block);
            fwrite(p.data(), 1, p.size(), stdout);
            return 0;
        }
        if (!jsonl.empty()) {   // one process, many questions: what scripts/conformance.py drives
            FILE * f = fopen(jsonl.c_str(), "rb");
            if (!f) { fprintf(stderr, "cannot open %s\n", jsonl.c_str()); return 1; }
            char * line = nullptr;
            size_t cap = 0;
            ssize_t n;
            while ((n = getline(&line, &cap, f)) > 0) {
                if (n <= 1) continue;
                const auto j = nlohmann::json::parse(line, nullptr, false);
                if (j.is_discarded()) { fprintf(stderr, "bad json line\n"); continue; }
                const auto q     = j.at("question");
                const auto type  = q.value("type", "noul");
                const auto instr = q.value("instructions", "");
                const auto st    = j.value("state", "");
                sense::Answer a;
                if (type == "noul") {
                    a = m.noul(st, instr);
                } else if (type == "choice") {
                    std::vector<std::pair<std::string, std::string>> opts;
                    for (auto it = q.at("criteria").begin(); it != q.at("criteria").end(); ++it)
                        opts.emplace_back(it.key(), it.value().get<std::string>());
                    a = m.choice(st, instr, opts, q.value("open_set", false));
                } else {
                    std::vector<std::string> levels;
                    for (const auto & l : q.at("criteria")) levels.push_back(l.get<std::string>());
                    a = m.score(st, instr, levels);
                }
                // the label may be a string or a number depending on the dataset
                const auto ans = j.contains("answer") ? (j["answer"].is_string() ? j["answer"].get<std::string>()
                                                                                 : j["answer"].dump())
                                                      : std::string();
                nlohmann::json out{{"type", a.type}, {"keys", a.keys}, {"p", a.probs},
                                   {"ms", a.us / 1000.0}, {"y", ans},
                                   {"task", j.value("task", "")}};
                printf("%s\n", out.dump().c_str());
                fflush(stdout);
            }
            free(line);
            fclose(f);
            return 0;
        }
        if (demo) {
            print(m.noul("部署连续失败两次，客户已经看到 500 错误。", "需要立即处理吗？"));
            print(m.choice("Checkout returns 502 for every EU customer since 09:00.", "Which team owns this?",
                           {{"infra", "Infrastructure"}, {"billing", "Billing"}}, true));
            print(m.score("Arrived two weeks late and the box was crushed.", "How satisfied is the customer?",
                          {"angry", "unhappy", "neutral", "happy"}));
            return 0;
        }
        if (!noul_q.empty())   print(m.noul(state, noul_q));
        if (!choice_q.empty()) print(m.choice(state, choice_q, options, open_set));
        if (!score_q.empty())  print(m.score(state, score_q, levels));
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
