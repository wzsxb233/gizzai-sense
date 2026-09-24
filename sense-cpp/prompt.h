// The text half of Sense-T's input: the series as words and digits, built the way the release builds it.
//
// A series reaches the model twice — as soft tokens from the encoder, and as text in the prompt. This is
// the text half. It has to match `release/sense_t.py` byte for byte: one digit of rounding difference is
// a different prompt, and therefore a different answer, with nothing in the output looking wrong.
#pragma once

#include <string>
#include <vector>

struct series_text {
    std::vector<std::string>         names;
    std::vector<std::vector<double>> values;   // one row per variate
    std::string                      unit;
    std::string                      step;
};

// Mean and standard deviation of the window the model reads. Chronos standardises the series away, so
// the actual levels only survive in the text.
std::string window_stats(const series_text & s, int window);

// What a dashboard would state: totals ("counts") or means ("levels") over 7 / 14 / 28 steps.
std::string series_summary(const series_text & s, const std::string & style);

// The read window as about `segments` segment means, plus the most recent values.
std::string numbers_long(const series_text & s, int window, int recent = 32, int segments = 96);

// state, context, the statistics, optionally the summary, optionally the digits, then the series slot.
std::string state_text(const series_text & s, const std::string & state, const std::string & context,
                       const std::string & mode, int window, int stats_window, const std::string & summary);
