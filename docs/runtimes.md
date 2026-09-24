# Running Sense elsewhere: vLLM, llama.cpp / Ollama, MLX, phones

Sense does not generate an answer, it **reads one**: build the prompt, run one forward pass, take the
logits at the last position, softmax over the answer's own option tokens. A runtime can host Sense only
if it gives back the probability of *specific* tokens. Everything below follows from that.

A port counts as supported only after `scripts/conformance.py` has put the same questions through it and
the reference, and the difference is measured: per-row |Δp|, how many answers flipped, accuracy and
calibration measured again. Running the weights and being Sense are different claims.

## Where each runtime stands (2026-09-23)

| runtime | architecture | probabilities of chosen tokens | status |
|---|---|---|---|
| transformers (reference) | native | logits in Python | the reference |
| **llama.cpp, in process** (`sense-cpp/`) | `Gemma4ForConditionalGeneration` in its converter | `llama_get_logits_ith`, ~250 lines on top | **working and measured**; the on-device path |
| **llama.cpp server** | same GGUF | grammar + `n_probs`, or top-N logprobs by token id | works, but see "the server is not the same model" |
| **Ollama** | same GGUF | **no** logprobs / grammar / logit_bias in its API | weights run; the calibrated readout does not |
| **vLLM** | `Gemma4ForConditionalGeneration`, text + image + video + audio | `allowed_token_ids` masks the rest before the softmax | client written; pending a free GPU |
| **MLX** | `gemma4.py` in mlx-lm | logits in Python | client written; **unverified — no Apple silicon here** |
| **Sense-T, transformers** | custom class, `auto_map` | — | **working**: `AutoModel.from_pretrained(..., trust_remote_code=True)` |
| **Sense-T, llama.cpp** | Chronos-2 encoder in ggml (`sense-cpp/series.*`) + Gemma 4 | the readout above | encoder exact to 1.7e-06, prompt text byte-identical on 200 cases, soft tokens splice in with **no engine change**; the pieces still need assembling into one binary |

## Measured: E2B, 240 held-out text questions, in-process C++ on CPU

| build | file | accuracy | median latency | mean \|Δp\| vs Q8_0 | answers flipped |
|---|---|---|---|---|---|
| Q8_0 | 4.7 GB | 0.733 | 780 ms | — | — |
| Q4_K_M | 3.2 GB | 0.725 | 529 ms | 0.058 (max 0.72) | **11 / 240 (4.6%)** |

The same comparison through the HTTP server, an independent implementation, gives 14 / 240 and mean 0.056:
**4-bit keeps accuracy and moves probabilities.** If thresholds are set on those probabilities, ship Q8_0
or re-fit the readout temperature for the build you ship.

## Six ways to get this wrong, all found by the conformance test

1. **Double BOS.** The transformers chat template emits `<bos>`; a server that also adds one answers a
   different prompt. It moved a three-way choice by 0.17 and looked exactly like quantisation drift.
2. **No BOS at all.** llama.cpp's templates deliberately leave BOS to the tokenizer. Porting the
   "send token ids, add_special=false" habit from fix 1 removed it entirely and weakened every answer.
   Rule: exactly one BOS, and prove it by comparing token ids with the reference.
3. **Scoring the wrong token.** With primer `Answer:`, the reference scores `" A"`, not `"A"`. A grammar
   of `"A" | "B" | "C"` silently scores different tokens: 0.026 off on a three-way choice.
4. **Temperatures left behind.** The release carries per-type temperatures (noul 1.185, choice 1.168,
   score 1.179). A port that defaults to 1.0 is differently calibrated, which is the one thing Sense sells.
5. **Undefined behaviour reading them.** Iterating `nlohmann::json`'s `.items()` on a temporary applied
   some temperatures and skipped others — the worst kind of bug, because the output still looks sane.
6. **Remote code that cannot import itself.** transformers copies only siblings it sees as
   `from .x import y`; `from . import x` is invisible to it, so `trust_remote_code=True` failed for
   everyone but us. The release now imports a name from every sibling it needs.

## The server is not the same model

The in-process path and llama.cpp's own HTTP server, on the same GGUF and byte-identical prompt tokens,
report different logits: about 0.19 nats on one three-way choice, which is 0.025 of probability. It is not
noise (the in-process path is bit-identical across thread counts), and it is not the slot count or prompt
caching (both tested). Prompt caching adds its own smaller shift (0.779 → 0.764 raw on the same question).

So: **measure conformance through the path you deploy**, and pin its configuration. The numbers above are
the in-process path, which is what the SDK and a phone would run.

## How to run

```bash
# llama.cpp, in process (the SDK)
cmake -B build -DLLAMA_CPP_DIR=/root/llama.cpp -DLLAMA_CPP_BUILD=/root/llama.cpp/build && cmake --build build -j
./build/sense-cli -m Gizzai-Sense-E2B-Q8_0.gguf --demo        # sense_readout.json beside it carries the temperatures

# llama.cpp server
llama-server -m Gizzai-Sense-E2B-Q8_0.gguf --port 8080
python runtimes/sense_llamacpp.py --endpoint http://127.0.0.1:8080 --demo

# vLLM
vllm serve Gizzai-Sense-E2B --served-model-name sense --port 8000
python runtimes/sense_vllm.py --endpoint http://127.0.0.1:8000 --demo

# MLX (Apple silicon)
python -m mlx_lm convert --hf-path Gizzai-Sense-E2B --mlx-path Sense-E2B-mlx -q --q-bits 4
python runtimes/sense_mlx.py --model Sense-E2B-mlx --tokenizer Gizzai-Sense-E2B --demo

# Sense-T, transformers
python -c "from transformers import AutoModel; m = AutoModel.from_pretrained('Gizzai-Sense-T-E2B', trust_remote_code=True)"
```

Conformance for any of them:

```bash
python scripts/conformance.py --runtime sensecpp --model Gizzai-Sense-E2B-Q4_K_M.gguf --out reports/q4.json
python scripts/conformance.py --runtime reference --model /root/release/Gizzai-Sense-E2B --out reports/ref.json
python scripts/conformance.py --compare reports/ref.json reports/q4.json
```

## Sense-T on other runtimes

Sense-T's input carries soft tokens from the frozen Chronos-2 encoder and its forecasts come from a head
on the last hidden state, so it needs a model class, not a client.

* **transformers — done.** `modeling_sense_t.py` ships in the release. `config.json` keeps
  `architectures: Gemma4ForConditionalGeneration`, so vLLM, llama.cpp and MLX still load the weights;
  `auto_map` adds the door that also brings up the encoder, projector and quantile head.
* **llama.cpp — the encoder runs, and it is exact.** `scripts/convert_series_gguf.py` packs Chronos-2
  (12 blocks, d_model 768, dual attention per block, ReLU FFN, RoPE) plus the projector and the quantile
  head into one 248 MB GGUF, and `sense-cpp/series.{h,cpp}` runs that graph in ggml. Raw numbers go in —
  standardise, arcsinh, patch, time-encode, embed, splice the register token, twelve blocks, project —
  and soft tokens come out, with no torch anywhere in the path.

  Chronos' own quantile forecast runs too (`series_forecast_base`, the head Sense-T's residual
  correction sits on): standardised quantiles match to 1.2e-06, and unscaled — sinh, then the row's own
  loc and scale — they track the test series' weekly cycle and trend with the levels never crossing.

  Checked stage by stage against the real encoder's own tensors (`scripts/dump_chronos_ref.py` saves
  every stage; `series-cli` compares them), on one fixed series with a covariate:

  | weights | worst block | soft tokens | file |
  |---|---|---|---|
  | F32 | 7.6e-06 | 1.7e-06 | 495 MB |
  | F16 | 1.2e-02 (1.0% of spread) | 6.5e-04 (0.05%) | 248 MB |

  F32 is float noise, so the graph is the reference graph. F16's per-element error looks worse in the
  middle of the stack than at the end because the projector's LayerNorm and the final normalisation
  absorb it: what reaches the language model is 0.05% off, far below anything that moves a decision.

  Three things the reference does that a port will not guess, all of which cost exact agreement:
  attention scores are **not** divided by √d_kv; group attention transposes time and batch, attending
  across variates at one patch with no RoPE; and the patch carries `[time encoding, values, mask]`,
  where the time encoding is `(i − C) / context_length` — not a position index.

  **And the soft tokens reach the language model with no engine change.** Gemma 4 gives every position
  a second, per-layer input looked up by token id, which is why feeding embeddings looked like it would
  need engine work. It does not, for this model: Sense-T puts its soft tokens in **PAD** slots and
  computes `per_layer_inputs` from those ids, and llama.cpp, decoding an embedding batch, substitutes
  `per_layer_tok_embd` row 0 — which is PAD, id 0. The two agree by construction. llama.cpp also does
  not apply Gemma's √n_embd embedding scale to supplied embeddings, matching transformers, where
  `get_input_embeddings()` has already applied it.

  Measured (`scripts/dump_splice_ref.py` + `sense-cpp/splice-cli.cpp`, E2B Q8_0, same prompt and the
  same soft tokens through both), as the probability of a two-way choice:

  | slots hold | llama.cpp | transformers | apart |
  |---|---|---|---|
  | their own PAD embeddings (control) | 0.1058 | 0.1011 | 0.0047 |
  | soft tokens, decoded as embeddings | 0.1766 | 0.1783 | **0.0017** |

  The soft tokens move the answer by +0.071 in llama.cpp and +0.077 in transformers. The splice agrees
  more closely than Q8_0's own error, so what remains is quantisation, not the mechanism. So Sense-T on
  llama.cpp is three decodes on one sequence — prefix tokens, soft-token embeddings, suffix tokens —
  and the readout already in `sense-cpp/`. Note this is verified for **PAD** slots; a model whose slots
  carry a dedicated token id (an image token, say) would need llama.cpp's row 0 to be that id instead,
  and llama.cpp's own comment marks that case as unverified.

  **The text half of the prompt is ported too.** A series reaches the model twice: as soft tokens, and
  as words and digits in the prompt. `sense-cpp/prompt.{h,cpp}` builds that text — the window's mean and
  standard deviation, the dashboard summary in either style, the read window as ~96 segment means plus
  the most recent values, and the assembly around the series slot. `scripts/dump_state_text.py` writes
  200 cases from the released Python (series shorter and longer than the window, lengths either side of
  the 96 segments, negatives, zeros, a flat row, values on rounding boundaries, one and two variates,
  every mode and summary style) and `prompt-cli` replays them: **200 of 200 byte-identical**, over
  163 kB of prompt text.

  Byte-identical is the only useful standard here, because a prompt that is nearly right is simply a
  different prompt and nothing downstream looks wrong. The check earns that claim — setting one
  parameter deliberately wrong (32 recent values to 31) breaks 144 of the 200 cases, so it can fail.

  **Assembled and run: `sense-cpp/sense-t-cli`.** A series and a question in, a calibrated probability
  out, with no Python anywhere. On a baked Sense-T release converted to a bf16 GGUF, 120 points become
  11 soft tokens, and the three-chunk decode answers in **941 ms on CPU**, against about 4 s for the
  released Python on the same machine. The prompt tokenises identically on both sides — 80 text tokens,
  11 soft, 35 text, the same option ids and primer — so the input is not in question.

  What differs is the engine. Measured on the same model and question:

  | | released Python (torch, bf16) | `sense-t-cli` (llama.cpp, bf16) | apart |
  |---|---|---|---|
  | the same question with no series at all | 0.5525 | 0.5348 | 0.018 |
  | with the series | 0.5927 | 0.5684 | 0.024 |

  The first row is the control, and it is the point: llama.cpp and PyTorch already answer 0.018 apart on
  this model *before any series is involved*, both times with llama.cpp lower, which is a precision
  offset rather than a structural fault. Everything the series contributes — encoder, projector, soft
  tokens, the splice — adds about 0.006 on top of that floor.

  So the port is faithful, and the honest statement is about the floor, not the port: a bf16 GGUF of
  this model is ~0.02 away from its PyTorch original, which matters if thresholds are set on these
  probabilities. Re-fit the readout temperature for the build being shipped, as with any quantisation.

  A seventh trap, found here: **a conformance test can pass because it is blind.** The first run of this
  comparison scaled the synthetic soft tokens to the norm of the embedding *weight* rows, which for
  Gemma are √1536 ≈ 39 times smaller than what the model actually reads. The perturbation was then so
  weak that quantisation noise dominated it, and the two sides differed by 0.025 with no way to tell
  whether that meant anything. Make the thing under test large enough to see before trusting agreement.
* **vLLM** — decisions could go through the prompt-embedding path, but Gemma 4 derives per-layer inputs
  from token ids, so that needs verifying before any claim. The quantile head needs the hidden state.
* **MLX** — needs Chronos-2 in MLX; not started.

## Phones

E2B at Q4_K_M is 3.2 GB: fits an 8 GB iPhone or Android flagship, tight on 6 GB (a Q3 build would cover
those). One engine covers both platforms — llama.cpp with Metal on iOS and Vulkan/CPU on Android — and
`sense-cpp/` is already that code: it reads the logits array directly, which is easier on a device than
over any HTTP API. Ship the model as a first-run download, not inside the app bundle.

Worth one test each: MLX Swift on iOS, LiteRT-LM / MediaPipe on Android for NPU. The deciding question is
the same one as everywhere else: does it expose token-level scores?
