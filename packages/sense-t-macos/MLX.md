# MLX bundle / MLX 本地测试包

The Apple bundle contains one copy of the released **GizzAI-Sense-E2B** checkpoint and one copy of the MLX readout adapter. The checkpoint is kept in Hugging Face format so conversion happens on the Mac with the local Apple Silicon toolchain. The UI still exposes all three models: E2B can be tested locally after conversion; E4B and Sense-T remain selectable through their matching HTTP services.

Apple 包包含一份 **GizzAI-Sense-E2B** 权重和一份 MLX readout adapter。权重保持 Hugging Face 格式，由你在 Mac 上使用 Apple Silicon 工具链转换。界面仍然提供三个模型选择：E2B 转换后可在本地测试；E4B 与 Sense-T 继续通过对应的 HTTP 服务测试。

## Convert and run / 转换并运行

```bash
cd mlx-bundle
python3 -m venv .venv
source .venv/bin/activate
pip install -U mlx mlx-lm transformers
python -m mlx_lm convert \
  --hf-path models/GizzAI-Sense-E2B \
  --mlx-path models/Sense-E2B-mlx \
  -q --q-bits 4
python runtimes/sense_mlx.py \
  --model models/Sense-E2B-mlx \
  --tokenizer models/GizzAI-Sense-E2B \
  --demo
```

Quantisation and Apple Silicon output still need to pass the repository conformance check before MLX is marked supported. / 量化与 Apple Silicon 输出仍需通过仓库一致性检查，之后才能把 MLX 标记为正式支持。
