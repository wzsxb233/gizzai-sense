"""Sense-T as a transformers model class.

    from transformers import AutoModel
    m = AutoModel.from_pretrained("Gizzai-Sense-T-E2B", trust_remote_code=True, dtype="bfloat16")
    out = m.decide(series, state="这家店最近在做促销。",
                   question={"type": "noul", "instructions": "下周会不会低于 800 件？"}, horizon=7)

The checkpoint stays a stock Gemma 4 checkpoint: `architectures` is unchanged, every tensor keeps its
name, and vLLM, llama.cpp and MLX load these weights as they always would. This class is an *additional*
entry point, reached through `auto_map`, that also brings up the parts a plain LM has no place for:

    chronos-2/      the frozen series encoder, bundled with the release (Apache 2.0, unmodified)
    projector.pt    the trained projector: Chronos states -> soft tokens in the LM's embedding space
    qhead.pt        the quantile head: last hidden state -> 10 / 50 / 90%, corrected from Chronos-2
    sense_t.json    how the inputs are rebuilt (mode, window, summaries), so a prompt is never guessed

Those three files sit beside the weights rather than inside them, so nothing a stock runtime reads has to
change; this class loads them from the model directory itself.

Copyright (c) 2026 邓颐村 (Deng Yicun). Licensed under the Gizzai Sense License; see LICENSE.
"""
from __future__ import annotations

import json
import os
from typing import Any, Sequence

import torch
from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer, PreTrainedModel

try:
    from .sense_format import FORMAT_ID as _FORMAT_ID  # noqa: F401  (makes the loader copy sense_format.py)
    from .sense_t import (LEVELS, QuantityHead, ResidualQuantityHead, SeriesEncoder, SenseT, _QuantityPlan,
                          numbers_long, series_norm, series_summary, window_stats)
    from . import sense_format as sf
except ImportError:                       # plain files next to each other, as shipped
    from sense_t import (LEVELS, QuantityHead, ResidualQuantityHead, SeriesEncoder, SenseT, _QuantityPlan,
                         numbers_long, series_norm, series_summary, window_stats)
    import sense_format as sf


try:      # the checkpoint keeps the base model's own config class, so AutoModel's registration check passes
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    _CONFIG_CLASS = CONFIG_MAPPING["gemma4"]
except Exception:                                                   # a transformers without Gemma 4
    _CONFIG_CLASS = None


class SenseTModel(PreTrainedModel):
    """Gemma 4 with a frozen series encoder, a trained projector and a quantile head.

    `forward` is the language model's forward, so anything that works on the base model works here.
    `decide` and `forecast` are the typed readout: one forward pass, no tokens generated.
    """

    config_class = _CONFIG_CLASS
    _supports_sdpa = True

    def __init__(self, config, model_dir: str | None = None, device_map: str | None = None, **kw):
        super().__init__(config)
        path = model_dir or getattr(config, "_name_or_path", None)
        if not path or not os.path.isdir(path):
            raise ValueError("SenseTModel needs the model directory (projector.pt, qhead.pt, sense_t.json, "
                             "chronos-2/ live beside the weights); pass model_dir=...")
        self.model_dir = path
        self.cfg_t = json.loads(open(os.path.join(path, "sense_t.json"), encoding="utf-8").read())
        if self.cfg_t.get("format") != sf.FORMAT_ID:
            raise ValueError(f"model expects prompt format {self.cfg_t.get('format')!r}; this file implements {sf.FORMAT_ID!r}")
        self.language_model = AutoModelForImageTextToText.from_pretrained(
            path, dtype=kw.get("dtype", torch.bfloat16), device_map=device_map or "cpu")
        dev = next(self.language_model.parameters()).device
        hidden = self.language_model.config.text_config.hidden_size
        chronos = os.path.join(path, "chronos-2")
        self.encoder = SeriesEncoder(hidden, dev, chronos if os.path.isdir(chronos) else self.cfg_t.get("chronos", "amazon/chronos-2"),
                                     ctx_patches=int(self.cfg_t.get("window", 256)) // 16)
        self.encoder.load_state_dict(torch.load(os.path.join(path, "projector.pt"), map_location=dev), strict=False)
        self.qhead = None
        if os.path.exists(os.path.join(path, "qhead.pt")):
            Head = ResidualQuantityHead if self.cfg_t.get("qty_head") == "residual" else QuantityHead
            self.qhead = Head(hidden, dev)
            self.qhead.load_state_dict(torch.load(os.path.join(path, "qhead.pt"), map_location=dev))
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self._readout = SenseT.__new__(SenseT)          # the released readout, over these modules
        self._readout.path, self._readout.cfg = path, self.cfg_t
        self._readout.mode = self.cfg_t.get("mode", "mixed")
        self._readout.window = int(self.cfg_t.get("window", 256))
        self._readout.stats_window = int(self.cfg_t.get("stats_window", 0)) or self._readout.window
        self._readout.summary = self.cfg_t.get("summary", "off")
        self._readout.temperature = self.cfg_t.get("temperature", {})
        self._readout.tok, self._readout.model = self.tokenizer, self.language_model
        self._readout.device = dev
        self._readout.pad_id = self.language_model.config.text_config.pad_token_id
        self._readout.encoder, self._readout.head = self.encoder, self.qhead
        self._readout._text = next(m for m in self.language_model.modules()
                                   if hasattr(m, "get_per_layer_inputs") and hasattr(m, "embed_tokens"))

    # ---------------------------------------------------------------- transformers surface
    @classmethod
    def from_pretrained(cls, path: str, *args, **kw):
        config = kw.pop("config", None) or AutoConfig.from_pretrained(path)
        config._name_or_path = path
        return cls(config, model_dir=path, device_map=kw.pop("device_map", None), **kw)

    def forward(self, *args, **kw):
        """The language model's own forward: this class adds inputs, it does not change the model."""
        return self.language_model(*args, **kw)

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    # ---------------------------------------------------------------- the typed readout
    @torch.no_grad()
    def decide(self, series: dict, question: dict, state: str = "", context: str = "", horizon: int = 8) -> dict:
        """One typed question about a series -> probabilities over its own options (nothing generated)."""
        return self._readout.ask(question, state, series, context=context, horizon=horizon)

    @torch.no_grad()
    def forecast(self, series: dict, step: int, horizon: int = 8, state: str = "", question: str | None = None) -> dict:
        """The value `step` ahead as 10 / 50 / 90% quantiles, in the series' own units."""
        return self._readout.forecast(series, step, horizon=horizon, state=state, question=question)

    def noul(self, series: dict, question: str, state: str = "", horizon: int = 8) -> float:
        return self.decide(series, {"type": "noul", "instructions": question}, state, horizon=horizon)["noul"]

    def choice(self, series: dict, question: str, options: dict[str, str], state: str = "",
               open_set: bool = False, horizon: int = 8) -> dict:
        return self.decide(series, {"type": "choice", "instructions": question, "criteria": options,
                                    "open_set": open_set}, state, horizon=horizon)

    def score(self, series: dict, question: str, levels: Sequence[str], state: str = "", horizon: int = 8) -> dict:
        return self.decide(series, {"type": "score", "instructions": question, "criteria": list(levels)},
                           state, horizon=horizon)
