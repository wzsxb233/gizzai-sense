# Sense family Android test package / Sense 系列 Android 试用包

A bilingual Jetpack Compose shell containing selectors for all three released models: `GizzAI-Sense-E2B`, `GizzAI-Sense-E4B`, and `GizzAI-Sense-T-E2B`. It uses the same GizzAI design tokens, colors, and brand mark as the Sense demos.

这是一个双语 Jetpack Compose 试用壳，包含三个已发布模型：`GizzAI-Sense-E2B`、`GizzAI-Sense-E4B` 与 `GizzAI-Sense-T-E2B`。它使用与 Sense demo 相同的 GizzAI design tokens、颜色与品牌标志。

## Run / 运行

Start the selected server on your computer / 在电脑启动所选模型服务:

```bash
python demos/live_server.py --model GizzAI-Sense-E2B --host 0.0.0.0 --port 8765
# or / 或
python demos/sense_t_server.py --model GizzAI-Sense-T-E2B --host 0.0.0.0 --port 8770
```

Open this folder in Android Studio and run the `app` target. In the emulator the default endpoint is `http://10.0.2.2:8770`; for Sense E2B/E4B change it to port `8765`; on a phone use the computer's LAN address. / 用 Android Studio 打开此目录并运行 `app`。模拟器默认地址为 `http://10.0.2.2:8770`；Sense E2B/E4B 改用 `8765` 端口；真机填写电脑局域网地址。

This is a UI/API test shell, not a claim that Android Vulkan, NPU, or native device conformance has passed. / 这是 UI/API 试用壳，不代表 Android Vulkan、NPU 或原生设备一致性已经通过；当前包保持 **TEST** 标记。
