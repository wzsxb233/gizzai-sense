# Sense-T: a bilingual technical report / Sense-T：中英双语技术报告

**Release note / 发布说明 — 2026-09-24**

This is the release-facing bilingual report for Gizzai-Sense-T-E2B. The longer research draft is
[`paper_sense_t.md`](https://github.com/wzsxb233/gizzai-sense/blob/main/docs/paper_sense_t.md).
/ 这是 Gizzai-Sense-T-E2B 的发布版双语技术报告；更长的研究草稿见
[`paper_sense_t.md`](https://github.com/wzsxb233/gizzai-sense/blob/main/docs/paper_sense_t.md)。

## Abstract / 摘要

Sense-T joins a frozen Chronos-2 time-series encoder to a small Gemma 4 language model. One request can
return a typed judgement, a calibrated probability over the requested options, and monotone 10/50/90%
forecast quantiles. The series is represented twice: as projected soft tokens in PAD slots and as a
compact digit summary in the prompt. The numeric head is residual to Chronos-2, so it remains the base
forecaster when the text contains no useful event.

Sense-T 将冻结的 Chronos-2 时序编码器连接到小型 Gemma 4 语言模型。一次请求可以返回类型化判断、请求选项上的校准概率，以及单调的 10/50/90% 预测分位数。时序有两种表示：投影到 PAD 槽位的 soft token，以及写入 prompt 的紧凑数字摘要。数值 head 以 Chronos-2 为 residual 基线，因此当文字没有提供有效事件时，它仍保持基础 forecaster 的行为。

## Model and readout / 模型与 readout

Chronos-2 is frozen. Each row is standardized, transformed with arcsinh, divided into patches, time-encoded,
and passed through Chronos attention. A trained projector maps the register and patch states into Gemma's
embedding scale. The language model then reads the requested option tokens at the final position; it does
not generate an explanation. A per-question-type temperature calibrates the option distribution.

Chronos-2 保持冻结。每一行数据先标准化、做 arcsinh 变换、切成 patch、加入时间编码，再经过 Chronos attention。训练得到的 projector 把 register 与 patch 状态映射到 Gemma 的 embedding 尺度。语言模型只读取最终位置上请求选项对应的 token，不生成解释文字。不同问题类型使用独立 temperature 校准选项分布。

For forecasts, the model predicts a correction to Chronos-2's quantiles: the median receives a shift and
quantile gaps are scaled positively, which preserves monotonicity. A text event can also be grounded into a
historical/future covariate and passed back to the forecaster.

对于预测，模型学习对 Chronos-2 分位数的修正：中位数加 shift，分位数间距做正值缩放，从而保持分位数单调。文字中的事件还可以经过 grounding，转成历史/未来 covariate，再传回 forecaster。

## Training and evaluation / 训练与评估

The training suite contains synthetic typed questions over anomalies, trends, thresholds, periods,
volatility, event effects, long history, and what-if interventions. Decision objectives use log loss and
Brier; quantile objectives use pinball loss. Results below are from the released all-suite evaluation.

训练数据包含关于异常、趋势、阈值、周期、波动、事件影响、长历史与 what-if 干预的合成时序类型化问题。判断使用 log loss 与 Brier loss，分位数预测使用 pinball loss。以下结果来自已发布的 all-suite 评估。

| Split / 数据划分 | Accuracy / 准确率 | Mean ECE / 平均 ECE |
|---|---:|---:|
| Held-out series, training domains / 留出序列、训练域 | **0.797** | **0.065** |
| Unseen domains / 未见域 | **0.771** | **0.077** |
| Unseen phrasings / 未见措辞 | **0.834** | **0.052** |

The released quantile head reports pinball 0.176 with 0.77 coverage on the held-out split, 0.238 with
0.70 coverage on unseen domains, and 0.179 with 0.73 coverage on unseen phrasings. The nominal 10–90%
band target is 0.80 coverage; the gaps are reported rather than hidden.

已发布的 quantile head 在留出划分上 pinball 为 0.176、覆盖率 0.77；未见域 pinball 为 0.238、覆盖率 0.70；未见措辞 pinball 为 0.179、覆盖率 0.73。名义上的 10–90% 区间目标覆盖率为 0.80，报告保留这些差距，不做隐藏。

## What the experiments show / 实验结论

The compact digit path is the reliable reader for local statistics at this model scale. Soft tokens retain
long context and seasonal shape, but they are harder for a small language model to read from limited
alignment data. Text-to-covariate grounding is the strongest practical fusion path: it lets the language
model read events and lets Chronos forecast them.

在当前模型规模下，紧凑数字路径更可靠地读取局部统计量。soft token 保留了长上下文与季节形状，但小型语言模型在有限 alignment 数据下较难读取它们。文字到 covariate 的 grounding 是最实用的融合路径：语言模型读取事件，Chronos 负责预测事件影响。

On an unseen real-data early-warning task, Sense-T was statistically level with a tuned gradient-boosting
baseline across two seeds; it did not beat that baseline. This negative result is part of the release claim.

在未参与训练的真实早期预警任务上，Sense-T 两个 seed 的结果与调优后的 gradient-boosting baseline 统计持平，并没有击败它。这个负结果也是发布结论的一部分。

## Runtime fidelity / 运行时一致性

The Transformers path is the reference and loads with `trust_remote_code=True`. The llama.cpp path is
measured: the Chronos graph's soft-token output is within 1.7e-6 at the checked stage, prompt text is
byte-identical on 200 generated cases, and the assembled Sense-T C++ CLI answers a case in about 941 ms on
CPU versus about 4 seconds for the released Python path. This is a measured build claim, not a claim that
every phone package is finished.

Transformers 路径是参考实现，可通过 `trust_remote_code=True` 加载。llama.cpp 路径已实测：Chronos 图在检查阶段的 soft-token 输出误差为 1.7e-6，200 个生成 case 的 prompt 文本字节级一致，组装后的 Sense-T C++ CLI 在 CPU 上约 941 ms 完成一个 case，而发布版 Python 约 4 秒。这是针对已测构建的声明，不代表所有手机包都已完成。

vLLM, Ollama and MLX remain experimental for Sense-T. They are not marked supported until the series
inputs, option-token probabilities and forecast head pass the same conformance test. / vLLM、Ollama 与 MLX 对 Sense-T 仍是实验性状态；在时序输入、选项 token 概率与 forecast head 通过同一套 conformance 测试前，不标记为已支持。

## Reproducibility / 复现

```bash
pip install -U huggingface_hub torch "transformers>=5.17" chronos-forecasting
hf download GizzAI/Gizzai-Sense-T-E2B --local-dir Gizzai-Sense-T-E2B
python - <<'PY'
from sense_t import SenseT
s = SenseT("Gizzai-Sense-T-E2B")
series = {"names": ["store sales"], "values": [[980, 1012, 1040, 995, 1030, 1105, 1220]], "unit": "units", "step": "day"}
print(s.forecast(series, step=2, horizon=7))
PY
```

The released demo video is a real-model recording. The bilingual edition carries Chinese
voice-over and dual Chinese-English captions. / 发布的视频是真实模型录制；双语版保留中文配音，并加入中英双语字幕。

## Limitations and use / 局限与使用

- Most training data is synthetic; the real-data result is a tie, not a win. / 大部分训练数据是合成的；真实数据结果是持平，不是获胜。
- Forecast bands cannot cover shocks absent from the input. / 输入没有提示的突发事件无法由预测区间保证覆盖。
- Long histories are truncated at the trained window. / 长历史会截断到训练窗口。
- Calibrated probabilities support decisions; they do not replace human review for material effects on people. / 校准概率用于辅助决策；对人产生重大影响的事项仍需人工复核。

## References / 参考文献

The full draft contains references to Chronos-2, LoRA, proper scoring rules, calibration, and the OULAD
dataset. / 完整草稿列出了 Chronos-2、LoRA、严格适当评分规则、校准与 OULAD 数据集的参考文献。

1. Ansari et al., *Chronos: Learning the Language of Time Series*, 2024.
2. Gneiting and Raftery, *Strictly Proper Scoring Rules, Prediction, and Estimation*, JASA, 2007.
3. Guo et al., *On Calibration of Modern Neural Networks*, ICML, 2017.
4. Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021.
5. Kuzilek et al., *Open University Learning Analytics dataset*, Scientific Data, 2017.
