# Sense demo Spaces / Sense 在线试用空间

The three model demos are published on both platforms. Each page is bilingual and uses the pinned
GizzAI design-system tokens and brand mark.

三个模型都已在两个平台发布双语试用页。每个页面都使用固定版本的 GizzAI design system token 与品牌标记。

| Model / 模型 | Hugging Face Space | ModelScope Studio | Runtime status / 运行状态 |
|---|---|---|---|
| Gizzai-Sense-E2B | [HF static hub](https://huggingface.co/spaces/GizzAI/Gizzai-Sense-E2B-Demo) | [ModelScope Studio](https://modelscope.cn/studios/GizzAI/Gizzai-Sense-E2B-Demo/summary) | page live; ModelScope Running on free 8 GB CPU / 页面可访问；魔搭免费 8 GB CPU 运行 |
| Gizzai-Sense-E4B | [HF static hub](https://huggingface.co/spaces/GizzAI/Gizzai-Sense-E4B-Demo) | [ModelScope Studio](https://modelscope.cn/studios/GizzAI/Gizzai-Sense-E4B-Demo/summary) | page live; ModelScope Running on free 8 GB CPU / 页面可访问；魔搭免费 8 GB CPU 运行 |
| Gizzai-Sense-T-E2B | [HF static hub](https://huggingface.co/spaces/GizzAI/Gizzai-Sense-T-E2B-Demo) | [ModelScope Studio](https://modelscope.cn/studios/GizzAI/Gizzai-Sense-T-E2B-Demo/summary) | page live; ModelScope Running on free 8 GB CPU / 页面可访问；魔搭免费 8 GB CPU 运行 |

## Quota / 配额

Hugging Face Gradio hardware creation returned HTTP 402 for the currently authenticated account and
`canPay=false`, so the HF pages are intentionally static bilingual hubs. They link to the model files and
the live ModelScope Studio. Static Spaces are public and free; they do not claim hosted inference.

当前登录的 Hugging Face 账号返回 `canPay=false`，创建 Gradio 硬件 Space 得到 HTTP 402，因此 HF 页面采用公开的
静态双语入口，链接到模型文件与可运行的魔搭创空间。静态 Space 免费公开，但不声称提供托管推理。

ModelScope currently exposes one free hardware option, `platform/2v-cpu-8g-mem`. GPU hardware is listed as
paid and requires the account owner's billing/quota. If the owner signs in with a quota-enabled account,
the existing Studio repositories can be switched to that hardware without recreating them.

魔搭当前只暴露一个免费硬件选项 `platform/2v-cpu-8g-mem`；GPU 硬件属于付费资源，需要账号所有者的计费或配额。
如果账号所有者改用有配额的登录账号，现有 Studio 不需要重建即可切换硬件。

## Reproducible source / 可复现源码

The bilingual Gradio source used by ModelScope is versioned in [`spaces/modelscope/`](../spaces/modelscope/),
and the static HF hubs are in [`spaces/huggingface/`](../spaces/huggingface/). The target-device shells are
versioned in [`packages/`](../packages/). They remain test shells until a macOS, iPhone or Android device
is available for physical-device conformance checks.

魔搭使用的双语 Gradio 源码已纳入 [`spaces/modelscope/`](../spaces/modelscope/)，HF 静态入口源码位于
[`spaces/huggingface/`](../spaces/huggingface/)。目标设备试用壳已纳入 [`packages/`](../packages/)。在拿到
macOS、iPhone 或 Android 真机并完成一致性检查前，它们仍标记为试用壳。
