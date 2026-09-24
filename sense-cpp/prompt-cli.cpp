// Replay every prompt-text case against the C++ port and compare byte for byte.
//
//   python scripts/dump_state_text.py --out /root/gguf/state_text_cases.json
//   ./build/prompt-cli --cases /root/gguf/state_text_cases.json
//
// The comparison is exact on purpose. "Close enough" prompt text is a different prompt.
#include "prompt.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

using json = nlohmann::json;

// where two strings first differ, with a little context, so a failure says what to fix
static void show_gap(const std::string & got, const std::string & want) {
    size_t i = 0;
    while (i < got.size() && i < want.size() && got[i] == want[i]) ++i;
    const size_t from = i > 40 ? i - 40 : 0;
    printf("      first differs at byte %zu of %zu (expected %zu)\n", i, got.size(), want.size());
    printf("      expected ...%s\n", want.substr(from, 90).c_str());
    printf("      got      ...%s\n", got.substr(from, 90).c_str());
}

int main(int argc, char ** argv) {
    const char * path = "/root/gguf/state_text_cases.json";
    int          show = 3;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--cases") && i + 1 < argc)      path = argv[++i];
        else if (!strcmp(argv[i], "--show") && i + 1 < argc)  show = atoi(argv[++i]);
    }

    std::ifstream f(path);
    if (!f) {
        fprintf(stderr, "[prompt] cannot open %s\n", path);
        return 1;
    }
    json cases;
    f >> cases;

    int bad = 0, n = 0;
    size_t bytes = 0;
    for (const auto & c : cases) {
        series_text s;
        for (const auto & nm : c["series"]["names"]) s.names.push_back(nm.get<std::string>());
        for (const auto & row : c["series"]["values"]) s.values.push_back(row.get<std::vector<double>>());
        s.unit = c["series"].value("unit", "");
        s.step = c["series"].value("step", "");

        const auto & st = c["settings"];
        const std::string got = state_text(s, c["state"].get<std::string>(), c["context"].get<std::string>(),
                                           st["mode"].get<std::string>(), st["window"].get<int>(),
                                           st["stats_window"].get<int>(), st["summary"].get<std::string>());
        const std::string want = c["expected"].get<std::string>();
        ++n;
        bytes += want.size();
        if (got != want) {
            if (bad < show) {
                printf("  ** %s\n", c["name"].get<std::string>().c_str());
                show_gap(got, want);
            }
            ++bad;
        }
    }
    printf("\n%d cases, %zu bytes of prompt text: %d match, %d differ\n", n, bytes, n - bad, bad);
    printf("%s\n", bad == 0 ? "the C++ prompt is byte-identical to the released Python."
                            : "the C++ prompt differs; the prompt is the input, so this must be exact.");
    return bad == 0 ? 0 : 1;
}
