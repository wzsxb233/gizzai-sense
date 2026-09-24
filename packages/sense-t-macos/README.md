# Sense family Apple test package / Sense 系列 Apple 试用包

This bilingual SwiftUI test shell contains selectors for all three released models: `GizzAI-Sense-E2B`, `GizzAI-Sense-E4B`, and `GizzAI-Sense-T-E2B`. It uses the GizzAI design tokens and calls the matching HTTP endpoint. The same source opens in Xcode for macOS and iPhone targets. It is an HTTP API test shell; it does not bundle a local MLX runtime or claim Apple Silicon conformance.

这是一个双语 SwiftUI 试用壳，包含三个已发布模型：`GizzAI-Sense-E2B`、`GizzAI-Sense-E4B` 与 `GizzAI-Sense-T-E2B`。它使用 GizzAI design tokens，并调用对应 HTTP 服务；同一份源码可在 Xcode 中构建 macOS 与 iPhone 目标。它是 HTTP API 试用壳，不内置本地 MLX runtime，也不代表已完成 Apple Silicon 一致性验证。

## Run / 运行

For Sense E2B/E4B, start the decision server / 对 Sense E2B/E4B 启动判断服务:

```bash
python demos/live_server.py --model GizzAI-Sense-E2B --port 8765
```

For Sense-T, start the time-series server / 对 Sense-T 启动时间序列服务:

```bash
python demos/sense_t_server.py --model GizzAI-Sense-T-E2B --port 8770
```

Open this folder in Xcode, choose the macOS or iPhone target, enter the endpoint, choose a model, and run a request. / 用 Xcode 打开此目录，选择 macOS 或 iPhone 目标，填写服务地址、选择模型并发起请求。

This remains a **TEST** package until real Mac, iPhone, Metal, and device conformance runs are recorded. / 在完成真实 Mac、iPhone、Metal 与设备一致性测试并记录前，本包保持 **TEST** 标记。

## MLX local test / MLX 本地测试

The Apple bundle includes one copy of the E2B checkpoint and the MLX adapter instructions in [`MLX.md`](MLX.md). The adapter is experimental until the Mac conformance run is completed. / Apple 包含一份 E2B 权重，并在 [`MLX.md`](MLX.md) 中提供 MLX adapter 说明；完成 Mac 一致性测试前，adapter 保持实验性。
