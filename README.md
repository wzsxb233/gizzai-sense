# Gizzai Sense · 叽喳 Sense

**Typed decisions with calibrated probabilities.** Ask a yes/no, pick-one or ordered-score question about
text, an image or a short audio clip, and get back a probability for every option you listed — nothing
else. The model never generates text for a decision: it reads the next-token distribution restricted to
one token per option, so the answer is always one of your options and the output is zero tokens.

This repository is the reference inference code. The weights are released separately:

| model | Hugging Face | ModelScope |
|---|---|---|
| Gizzai-Sense-E2B (small, for edge devices) | [GizzAI/Gizzai-Sense-E2B](https://huggingface.co/GizzAI/Gizzai-Sense-E2B) | [GizzAI/Gizzai-Sense-E2B](https://modelscope.cn/models/GizzAI/Gizzai-Sense-E2B) |
| Gizzai-Sense-E4B (stronger, for servers) | [GizzAI/Gizzai-Sense-E4B](https://huggingface.co/GizzAI/Gizzai-Sense-E4B) | [GizzAI/Gizzai-Sense-E4B](https://modelscope.cn/models/GizzAI/Gizzai-Sense-E4B) |

Live bilingual demo hubs / 在线双语试用入口: [E2B on HF](https://huggingface.co/spaces/GizzAI/Gizzai-Sense-E2B-Demo) · [E2B on ModelScope](https://modelscope.cn/studios/GizzAI/Gizzai-Sense-E2B-Demo/summary) · [E4B on HF](https://huggingface.co/spaces/GizzAI/Gizzai-Sense-E4B-Demo) · [E4B on ModelScope](https://modelscope.cn/studios/GizzAI/Gizzai-Sense-E4B-Demo/summary). The full Space and quota status is in [`docs/spaces.md`](docs/spaces.md) / 完整 Space 与配额状态见 [`docs/spaces.md`](docs/spaces.md)。

Technical report (English and Chinese, with demo videos):
[GizzAI/Gizzai-Sense-Report](https://modelscope.cn/studios/GizzAI/Gizzai-Sense-Report)

Both models are built on Google's Gemma 4 (`google/gemma-4-E2B-it`, `google/gemma-4-E4B-it`). The weights
are merged, so they load with plain `transformers`; no adapter library is needed.

## Quick start

```bash
pip install -r requirements.txt
hf download GizzAI/Gizzai-Sense-E2B --local-dir Gizzai-Sense-E2B
# or: modelscope download --model GizzAI/Gizzai-Sense-E2B --local_dir Gizzai-Sense-E2B
```

```python
from sense import Sense

s = Sense("Gizzai-Sense-E2B")          # the downloaded folder

print(s.noul("部署连续失败两次，客户已经看到 500 错误。", "需要立即处理吗？"))          # P(yes)
print(s.choice("Checkout returns 502 for every EU customer since 09:00.", "Which team owns this?",
               {"infra": "Infrastructure", "billing": "Billing"}, open_set=True))   # adds "none of these"
print(s.score("Arrived two weeks late and the box was crushed.", "How satisfied is the customer?",
              ["angry", "unhappy", "neutral", "happy"]))
print(s.noul("", "Is there a dog in the photo?", media=[{"type": "image", "image": "photo.jpg"}]))
```

`examples/quickstart.py` does the same end to end, including the download.

## API

| call | returns |
|---|---|
| `Sense(path, device="cuda", dtype=torch.bfloat16)` | loads the weights and the calibration in `sense_readout.json` |
| `s.noul(state, question, media=())` | `P(yes)` as a float |
| `s.choice(state, question, options, open_set=False, media=())` | `{"choice", "probabilities", "confidence", ...}`; `options` maps key → description; `open_set=True` adds `"__other__"` ("none of these") |
| `s.score(state, question, levels, media=())` | `{"score" (expected level), "probabilities", "confidence", ...}`; `levels` ordered low → high |
| `s.ask(question_dict, state="", media=())` | the same answer dict for `{"type": "noul" \| "choice" \| "score", "instructions", "criteria"}` |
| `s.decide(state, {id: question_dict})` | several independent questions about one state |

`state` is any text (or a list of strings, or JSON-serialisable data). `media` items look like
`{"type": "image", "image": "photo.jpg"}` or `{"type": "audio", "audio": "clip.wav"}`; audio is limited
to 30 s per clip (the base model's native limit).

**Probabilities are temperature-calibrated.** `confidence` is 1 minus the normalised entropy (`|2p − 1|`
for yes/no): 0 means the options are equally likely and 1 means one is certain. To route uncertain
answers to a person, compare your threshold with the top value in `probabilities`, not with
`confidence`.

## Scope of this code

This is the reference implementation: one question per forward pass, no batching or cache sharing, and
media within the base model's native limits. It reproduces the published evaluation numbers. Optimised
serving (shared-prefix batching, long audio and video, multi-step decision graphs, open-set generation),
the training and RL frameworks, and commercial or academic collaboration: **dengyicun@gizzai.com**.

Tested with torch 2.14 and transformers 5.17. `pillow` is needed for images and `librosa` (or
`torchcodec`) for audio files.

## Intended use

Gizzai Sense is built to make fast, calibrated judgements that software can act on, and to hand
uncertain cases to a stronger model or a person. It is **not** for fully automated decisions with legal
or similarly significant effects on people: keep a human in the loop for those. See `USE_POLICY.md`.

## License

The code and the weights are licensed under the **Gizzai Sense License v1.1**: the Apache License 2.0
(`LICENSE-APACHE-2.0`) plus additional terms (`LICENSE`). Individuals, academic research and
organisations with annual revenue under US$100,000 may use it free of charge; for commercial licensing,
write to dengyicun@gizzai.com. Use of the Gemma-derived weights is also subject to Google's Gemma terms
(see `NOTICE`).

© 2026 GizzAI Technology Limited Co. · 广州叽喳科技有限公司

---

## 中文简介

叽喳 Sense 输出带校准概率的类型化判断（是/否、单选、评分），支持文本、图像和 30 秒以内的音频。模型不生成任何
文字，只在你给出的选项上输出概率，答案一定落在选项之内，输出 token 数为零。本仓库是参考推理代码；模型权重
在 Hugging Face 与魔搭社区发布（见上表），中英双语技术报告见
[魔搭创空间](https://modelscope.cn/studios/GizzAI/Gizzai-Sense-Report)。

许可协议为 Apache 2.0 加附加条款（见 `LICENSE`）：个人、学术研究与年营收 10 万美元以下的机构可免费使用。
商业授权、学术合作，以及高性能推理、训练与强化学习框架，请联系 dengyicun@gizzai.com。

## Sense-T time-series inference / Sense-T 时序推理

The Sense-T release adds typed judgements and calibrated 10/50/90% forecasts over a time series. Its
reference implementation is [`sense_t.py`](sense_t.py), with the Transformers remote-code adapter in
[`modeling_sense_t.py`](modeling_sense_t.py). The public checkpoint is the compensated-baked Sense-T artifact; its bundled series runtime files are published alongside it, so users do not merge a separate LoRA adapter. The model is published at [GizzAI/Gizzai-Sense-T-E2B](https://huggingface.co/GizzAI/Gizzai-Sense-T-E2B).

Sense-T 在时序上增加类型化判断，以及经过校准的 10/50/90% 预测。参考实现是
[`sense_t.py`](sense_t.py)，Transformers remote-code 适配器是 [`modeling_sense_t.py`](modeling_sense_t.py)。公开 checkpoint 是完成 compensated bake 的 Sense-T 产物；时序运行所需文件随模型一并发布，用户无需再合并单独的 LoRA adapter。模型发布在
[GizzAI/Gizzai-Sense-T-E2B](https://huggingface.co/GizzAI/Gizzai-Sense-T-E2B)。

### Runtime status / 运行时状态

| Runtime / 运行时 | Status / 状态 |
|---|---|
| Transformers | **Verified / 已验证** |
| llama.cpp | **Verified measured C++ path / 已验证的实测 C++ 路径** |
| Ollama | **GGUF smoke test only; calibrated Sense readout not conformed / 仅 GGUF 冒烟测试，校准 Sense readout 未完成一致性验证** |
| vLLM | **Adapter experimental; no endpoint conformance result / 适配器实验性，尚无 endpoint conformance 结果** |
| MLX | **Adapter experimental; requires Apple Silicon verification / 适配器实验性，需要 Apple Silicon 验证** |
| Android / iPhone / macOS | **Bilingual test shells; no physical-device claim yet / 双语试用壳，尚无真机完成声明** |

The measured runtime evidence and reproduction commands are in [`docs/runtimes.md`](docs/runtimes.md).
/ 实测证据与复现命令见 [`docs/runtimes.md`](docs/runtimes.md)。

The bilingual Sense-T technical report is [`docs/paper_sense_t_bilingual.md`](docs/paper_sense_t_bilingual.md).
/ Sense-T 中英双语技术报告见 [`docs/paper_sense_t_bilingual.md`](docs/paper_sense_t_bilingual.md)。

### Target-device packages / 目标设备试用包

- [`packages/sense-t-macos/`](packages/sense-t-macos/) — SwiftUI bilingual macOS + iPhone shell with all three model selectors / SwiftUI 双语 macOS + iPhone 三模型选择试用壳
- [`packages/sense-t-android/`](packages/sense-t-android/) — Jetpack Compose bilingual Android shell with all three model selectors / Jetpack Compose 双语 Android 三模型选择试用壳

Both use the existing GizzAI design-system tokens, fonts and brand assets, and connect to the released
Sense family APIs. They require a running matching server; they are test packages until the target devices pass
runtime conformance. / 两个包都使用现有 GizzAI design system 的 token、字体和品牌资源，并连接三个模型对应的 API。它们需要运行中的匹配服务；目标设备通过运行时一致性验证前，保持为试用包。
