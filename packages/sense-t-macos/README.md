# Sense-T macOS test package / Sense-T macOS 试用包

This is a bilingual SwiftUI shell using the GizzAI design tokens and brand mark. It calls the released
Sense-T HTTP API so you can test the UI on a Mac before native Metal packaging is validated.

这是一个使用 GizzAI design tokens 与品牌标志的双语 SwiftUI 试用壳。它调用已发布的 Sense-T HTTP API，便于在 Mac 上先试用界面；原生 Metal 打包仍需在 Mac 上验证。

## Run / 运行

1. Start the server / 启动服务:

```bash
python demos/sense_t_server.py --model GizzAI-Sense-T-E2B --port 8770
```

2. On macOS, open this folder in Xcode, or run `swift run` from Terminal. / 在 macOS 用 Xcode 打开此目录，或在终端执行 `swift run`。

3. Enter the API endpoint and press Forecast. / 填写 API 地址并点击“预测”。

The sample sends `/v1/forecast` and displays the model's P10/P50/P90 response. This package is marked
**TEST** until a real Mac build and Metal inference conformance run are recorded.

示例调用 `/v1/forecast`，显示模型返回的 P10/P50/P90。完成真实 Mac 构建与 Metal 推理一致性测试并记录后，才会移除 **TEST** 标记。
