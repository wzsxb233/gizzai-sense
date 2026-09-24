# MLX all-model bundle / MLX 三模型包

The full Apple bundle contains all released model assets without duplicating the shared base checkpoint:

- `models/GizzAI-Sense-E2B/` — one copy of the E2B checkpoint and its readout files.
- `models/GizzAI-Sense-E4B/` — the separate E4B checkpoint.
- `models/GizzAI-Sense-T-E2B-adapter/` — Sense-T's series adapter, quantile head, and bundled Chronos encoder. Sense-T reuses the E2B base checkpoint above; its base weight shards are not copied a second time.
- `runtimes/sense_mlx.py` — the MLX typed-readout adapter.

完整 Apple 包包含全部已发布模型资源，并避免重复复制共享基础权重：

- `models/GizzAI-Sense-E2B/`：一份 E2B 权重与 readout 文件。
- `models/GizzAI-Sense-E4B/`：独立的 E4B 权重。
- `models/GizzAI-Sense-T-E2B-adapter/`：Sense-T 的时序 adapter、quantile head 与内置 Chronos encoder。Sense-T 复用上面的 E2B 基础权重，不再复制一套分片。
- `runtimes/sense_mlx.py`：MLX 类型化 readout adapter。

## Convert and run Sense E2B / E4B / 转换并运行 Sense E2B / E4B

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

Use the same commands with `models/GizzAI-Sense-E4B` for E4B. The current MLX readout adapter covers the base Sense decision path. The Sense-T series adapter assets are included for the Mac-side implementation, but Sense-T MLX remains experimental until its series path and forecast head pass conformance on Apple Silicon. / E4B 可将命令中的目录替换为 `models/GizzAI-Sense-E4B`。当前 MLX readout adapter 覆盖基础 Sense 判断路径；Sense-T 时序 adapter 资源已随包提供，但在 Apple Silicon 上完成时序路径与 forecast head 一致性测试前，仍保持实验性。
