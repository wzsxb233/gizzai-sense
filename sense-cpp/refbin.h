// A reference dump written by scripts/ref_to_bin.py or scripts/dump_splice_ref.py:
//     magic "SREF", uint32 count, then per tensor:
//         uint32 name_len, name, uint32 ndim, uint32 dims[ndim], float32 data[prod(dims)]
// Everything is float32, including token ids, which are exact well past this vocabulary.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

struct ref_tensor {
    std::vector<uint32_t> dims;
    std::vector<float>    data;
};

inline bool ref_load(const char * path, std::map<std::string, ref_tensor> & out) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[ref] cannot open %s\n", path);
        return false;
    }
    char     magic[4];
    uint32_t n = 0;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "SREF", 4) != 0 || fread(&n, 4, 1, f) != 1) {
        fprintf(stderr, "[ref] %s is not a reference dump\n", path);
        fclose(f);
        return false;
    }
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t len = 0, nd = 0;
        if (fread(&len, 4, 1, f) != 1) break;
        std::string name(len, '\0');
        if (fread(&name[0], 1, len, f) != len) break;
        if (fread(&nd, 4, 1, f) != 1) break;
        ref_tensor t;
        t.dims.resize(nd);
        if (fread(t.dims.data(), 4, nd, f) != nd) break;
        size_t ne = 1;
        for (uint32_t d : t.dims) ne *= d;
        t.data.resize(ne);
        if (fread(t.data.data(), 4, ne, f) != ne) break;
        out[name] = std::move(t);
    }
    fclose(f);
    return true;
}
