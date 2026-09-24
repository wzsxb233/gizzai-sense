#include "prompt.h"

#include <cmath>
#include <cstdio>

static const char * SERIES_MARK = "[[SERIES]]";

// printf and Python's format both round the same double correctly, so one decimal here is one decimal
// there. Everything that reaches the prompt goes through this, so there is one place to be wrong.
static std::string fmt(double v, int places) {
    char b[64];
    snprintf(b, sizeof(b), "%.*f", places, v);
    return b;
}

// python's v[-keep:]  (keep <= 0 means the whole row, matching `_arr(v)` with no keep)
static std::vector<double> tail(const std::vector<double> & v, int keep) {
    if (keep <= 0 || (size_t) keep >= v.size()) return v;
    return std::vector<double>(v.end() - keep, v.end());
}

// python's v[a:b] for negative a and b, which clamp rather than wrap
static std::vector<double> span(const std::vector<double> & v, int a, int b) {
    const int n = (int) v.size();
    int lo = a < 0 ? n + a : a, hi = b < 0 ? n + b : b;
    lo = lo < 0 ? 0 : (lo > n ? n : lo);
    hi = hi < 0 ? 0 : (hi > n ? n : hi);
    if (hi <= lo) return {};
    return std::vector<double>(v.begin() + lo, v.begin() + hi);
}

static double sum_of(const std::vector<double> & v) {
    double s = 0.0;
    for (double x : v) s += x;
    return s;
}

static double mean_of(const std::vector<double> & v) {
    return v.empty() ? NAN : sum_of(v) / (double) v.size();   // numpy gives nan on an empty mean too
}

static double std_of(const std::vector<double> & v) {
    if (v.empty()) return NAN;
    const double m = mean_of(v);
    double acc = 0.0;
    for (double x : v) acc += (x - m) * (x - m);
    return std::sqrt(acc / (double) v.size());                // population s.d., as numpy defaults to
}

std::string window_stats(const series_text & s, int window) {
    std::string out;
    for (size_t i = 0; i < s.names.size() && i < s.values.size(); ++i) {
        const std::vector<double> w = tail(s.values[i], window);
        if (i) out += "；";
        out += s.names[i] + "最近 " + std::to_string(w.size()) + " 个点：均值 " + fmt(mean_of(w), 1)
             + "，标准差 " + fmt(std_of(w), 1);
    }
    return out + "。";
}

std::string series_summary(const series_text & s, const std::string & style) {
    const std::string step = s.step.empty() ? "步" : s.step;
    std::string out;
    for (size_t i = 0; i < s.names.size() && i < s.values.size() && i < 3; ++i) {   // a dashboard shows a few
        const std::vector<double> & x = s.values[i];
        const std::vector<double> w7 = span(x, -7, (int) x.size()), w14 = span(x, -14, (int) x.size());
        const std::vector<double> p14 = span(x, -28, -14), w28 = span(x, -28, (int) x.size());
        if (style == "counts") {
            long since = (long) x.size();
            for (long j = (long) x.size() - 1; j >= 0; --j) {
                if (x[j] > 0.0) { since = (long) x.size() - 1 - j; break; }
            }
            long active = 0;
            for (double v : w14) active += v > 0.0 ? 1 : 0;
            out += s.names[i] + "：最近 7 " + step + "共 " + fmt(sum_of(w7), 0) + s.unit
                 + "，最近 14 " + step + "共 " + fmt(sum_of(w14), 0) + s.unit
                 + "，再往前 14 " + step + "共 " + fmt(sum_of(p14), 0) + s.unit
                 + "，最近 28 " + step + "共 " + fmt(sum_of(w28), 0) + s.unit
                 + "；最近 14 " + step + "里有 " + std::to_string(active) + " " + step
                 + "有活动，距上一次活动已过 " + std::to_string(since) + " " + step + "。";
        } else {
            out += s.names[i] + "：最近 7 " + step + "均值 " + fmt(mean_of(w7), 1) + s.unit
                 + "，最近 14 " + step + "均值 " + fmt(mean_of(w14), 1) + s.unit
                 + "，再往前 14 " + step + "均值 " + fmt(mean_of(p14), 1) + s.unit
                 + "，最近 28 " + step + "均值 " + fmt(mean_of(w28), 1) + s.unit + "。";
        }
    }
    return out;
}

std::string numbers_long(const series_text & s, int window, int recent, int segments) {
    std::string out;
    for (size_t i = 0; i < s.names.size() && i < s.values.size(); ++i) {
        const std::vector<double> w = tail(s.values[i], window);
        const int n = (int) w.size();
        int seg = (n + segments - 1) / segments;      // ceil(n / segments)
        if (seg < 1) seg = 1;
        if (i) out += "\n";
        out += s.names[i] + " 全部 " + std::to_string(n) + " 个点，每 " + std::to_string(seg)
             + " 个点的平均值（从早到晚）：";
        for (int j = 0; j < n; j += seg) {
            const int hi = j + seg > n ? n : j + seg;   // the last segment is short when it does not divide
            out += (j ? ", " : "") + fmt(mean_of(std::vector<double>(w.begin() + j, w.begin() + hi)), 1);
        }
        const std::vector<double> last = tail(w, recent);
        out += "\n" + s.names[i] + " 最近 " + std::to_string(last.size()) + " 个值：";
        for (size_t j = 0; j < last.size(); ++j) out += (j ? ", " : "") + fmt(last[j], 1);
    }
    return out;
}

std::string state_text(const series_text & s, const std::string & state, const std::string & context,
                       const std::string & mode, int window, int stats_window, const std::string & summary) {
    std::string base;                                  // "\n\n".join of whichever of the two is non-empty
    if (!state.empty()) base = state;
    if (!context.empty()) base += (base.empty() ? "" : "\n\n") + context;
    base += "\n" + window_stats(s, stats_window ? stats_window : window);
    if (summary != "off") base += "\n" + series_summary(s, summary);
    const std::string digits = mode == "mixed" ? numbers_long(s, window) + "\n" : "";
    return base + "\n" + digits + SERIES_MARK + "\n";
}
