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
