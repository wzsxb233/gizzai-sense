# Gizzai Sense runtime verification / Gizzai Sense 运行时验证

This document separates source availability from a measured runtime claim. A runtime is **verified** only
when the same questions go through the reference and the target path, option-token probabilities are
compared, and the result is recorded. / 本文把“有适配代码”和“已经验证”分开。只有同一组问题同时经过参考实现与目标运行时、比较选项 token 概率并留下结果，才算“已验证”。

## Status / 状态

| Runtime / 运行时 | What is present / 已有内容 | Verification / 验证 | Release status / 发布状态 |
|---|---|---|---|
| Transformers | Native Sense and Sense-T remote code / 原生 Sense 与 Sense-T remote code | **Verified**: model load, parity and Sense-T bake checks passed / **已验证**：模型加载、parity、Sense-T bake 均通过 | Supported / 支持 |
| llama.cpp | In-process C++ readout, series encoder, Sense-T CLI / 进程内 C++ readout、时序编码器、Sense-T CLI | **Verified**: encoder max error 1.7e-6, prompt 200/200 byte-identical; Sense-T CLI measured on CPU / **已验证**：编码器最大误差 1.7e-6，prompt 200/200 字节一致，Sense-T CLI 已完成 CPU 实测 | Supported for the measured build / 对已测构建支持 |
| Ollama | GGUF import and API smoke test / GGUF 导入与 API 冒烟测试 | **Not conformed**: Ollama 0.34.4 imports the GGUF; `/v1/chat/completions` returns logprobs, but the exact Sense prompt/token path and calibrated readout have not passed conformance / **未完成一致性验证**：Ollama 0.34.4 可导入 GGUF，`/v1/chat/completions` 能返回 logprobs，但 Sense 的精确 prompt/token 路径与校准 readout 尚未通过 conformance | Experimental; do not call supported / 实验性，不标记为支持 |
| vLLM | Adapter and Gemma-4 model support are present / 有 adapter，当前 vLLM 有 Gemma-4 模型支持 | **Not verified here**: the available vLLM wheel requires a separate dependency stack (including pinned torch/CUDA packages); no endpoint conformance result was produced / **本机未验证**：可用 vLLM wheel 需要独立依赖栈（含固定 torch/CUDA 包），尚未产出 endpoint conformance 结果 | Experimental / 实验性 |
| MLX | Adapter source / 适配器源码 | **Unavailable on this x86_64 host**: no Apple Silicon runtime was available / **本机不可验证**：当前机器是 x86_64，没有 Apple Silicon MLX 环境 | Experimental; Mac package is for target-machine testing / 实验性；macOS 包用于目标机测试 |
| Android | Portable C++ core and Compose test shell / 可移植 C++ 核心与 Compose 试用壳 | **No physical-device result yet**: no Android SDK/device in this environment / **尚无真机结果**：当前环境没有 Android SDK/设备 | Test package only / 试用包 |
| iPhone / macOS | llama.cpp design path and SwiftUI test shell / llama.cpp 路径与 SwiftUI 试用壳 | **No Apple-device result yet**: no Apple toolchain/device here / **尚无 Apple 设备结果**：当前环境没有 Apple 工具链/设备 | Test package only / 试用包 |

## Reproduce the verified paths / 复现已验证路径

```bash
# Transformers / Transformers
python -c "from transformers import AutoModel; AutoModel.from_pretrained('GizzAI/Gizzai-Sense-T-E2B', trust_remote_code=True)"

# llama.cpp in-process / llama.cpp 进程内路径
cmake -B build -DLLAMA_CPP_DIR=/path/to/llama.cpp -DLLAMA_CPP_BUILD=/path/to/llama.cpp/build
cmake --build build -j
./build/sense-t-cli --help
```

The C++ path is the measured device core. Metal on macOS/iPhone and Vulkan/CPU on Android still need
an on-device build and test. / C++ 路径是已经测过的设备核心；macOS/iPhone 的 Metal 与 Android 的 Vulkan/CPU 仍需要在目标设备上构建和测试。

## Sense-T limits / Sense-T 限制

Sense-T sends Chronos-2 soft tokens plus a calibrated quantile head. A text-only server adapter cannot
be called a Sense-T implementation until it accepts that series path and returns the same forecasts.
The current vLLM and MLX adapters are for the base text model; Sense-T on those runtimes is not claimed.
/ Sense-T 会发送 Chronos-2 soft tokens，并使用校准的 quantile head。仅支持文本的 server adapter 在接入时序路径并返回一致预测之前，不能称为 Sense-T 实现。当前 vLLM 与 MLX adapter 针对基础文本模型，未宣称支持 Sense-T。

## Test packages / 试用包

The repository contains two bilingual target-device shells. They use the same `--gz-*` design tokens,
GizzAI fonts and brand mark as the demos, and call the released Sense `/v1/decide` or Sense-T `/v1/forecast` API. / 仓库中有两个双语目标设备试用壳，使用 demo 相同的 `--gz-*` 设计 token、GizzAI 字体与品牌标志，并调用已发布 Sense 的 `/v1/decide` 或 Sense-T 的 `/v1/forecast` API。

- `packages/sense-t-macos/` — SwiftUI / macOS + iPhone 三模型试用包
- `packages/sense-t-android/` — Jetpack Compose / Android 三模型试用包

Both packages expose all three model selectors and require a matching server. They remain test shells until a real Mac, iPhone and Android device pass the runtime checklist. / 两个包都提供三个模型选择，并需要匹配的服务；在真实 Mac、iPhone 与 Android 设备通过运行时清单前，它们保持“试用壳”状态。
