# Sense-T Android test package / Sense-T Android 试用包

A bilingual Jetpack Compose shell using the same GizzAI design tokens, colors and brand mark as the
Sense demos. It calls the released Sense-T `/v1/forecast` API and is intentionally marked **TEST**.

这是一个双语 Jetpack Compose 试用壳，使用与 Sense demo 相同的 GizzAI design tokens、颜色与品牌标志。它调用已发布的 Sense-T `/v1/forecast` API，并明确标记为 **TEST**。

## Run / 运行

1. Start the server on your computer / 在电脑启动服务:

```bash
python demos/sense_t_server.py --model GizzAI-Sense-T-E2B --host 0.0.0.0 --port 8770
```

2. Open this folder in Android Studio and run the `app` target. In the emulator the default endpoint is
`http://10.0.2.2:8770`; on a phone use the computer's LAN address. / 用 Android Studio 打开此目录并运行 `app`。模拟器默认使用 `http://10.0.2.2:8770`；真机填写电脑局域网地址。

This is a UI/API test shell, not a claim that the model has passed Android Vulkan or NPU conformance.
The native llama.cpp device build will be added after a real Android SDK/device test.

这是 UI/API 试用壳，不代表模型已经通过 Android Vulkan 或 NPU 一致性验证。真实 Android SDK/设备测试完成后，再接入原生 llama.cpp 设备构建。
