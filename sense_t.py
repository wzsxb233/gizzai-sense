"""Gizzai Sense-T: a time series in, a calibrated typed answer out.

    from sense_t import SenseT
    s = SenseT("Gizzai-Sense-T-E2B")
    series = {"names": ["门店A 日销量"], "values": [[...]], "unit": "件", "step": "天"}

    s.noul("这家店最近在做促销。", "下一周销量会不会低于 800 件？", series, horizon=7)
    s.choice("", "未来 7 天的走势？", {"上升": "上升", "持平": "持平", "下降": "下降"}, series, horizon=7)
    s.forecast(series, step=2, horizon=7)          # 10 / 50 / 90% quantiles in the series' own units

The series is read by a frozen Chronos-2 encoder (Amazon, Apache-2.0, included in this repository): one
state per 16-step patch, a summary state, and future-patch states carrying its forecast. A small trained
projector maps those states into the language model's embedding space, where they sit in slots inside the
<state> block -- the same mechanism Gemma 4 uses for image and audio. Values are also written into the
prompt as digits, so the model can read a level exactly and the shape through the encoder.

Decisions come back as calibrated probabilities over the answer's own options (no text is generated), and
numbers come back as three monotone quantiles corrected from Chronos-2's own forecast, so a band means
what it says. Copyright (c) 2026 邓颐村 (Deng Yicun). Licensed under the Gizzai Sense License; see LICENSE.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

try:
    # transformers' remote-code loader copies siblings it sees as `from .x import y`; the explicit name
    # below is what makes it copy sense_format.py, so `trust_remote_code=True` works from a hub repo
    from .sense_format import FORMAT_ID as _FORMAT_ID  # noqa: F401
    from . import sense_format as sf
    from .sense import _answer           # one answer shape for text and series decisions
except ImportError:                       # used as plain files next to each other
    import sense_format as sf
    from sense import _answer

SERIES_MARK = "[[SERIES]]"
LEVELS = (0.1, 0.5, 0.9)


# ----------------------------------------------------------------- the series, as text
def _arr(values, keep: int | None = None) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    return v[-keep:] if keep else v


def window_stats(series: dict, window: int) -> str:
    """Mean and s.d. of the window the model reads: Chronos normalises the series, so levels live in text."""
    parts = []
    for name, v in zip(series["names"], series["values"]):
        w = _arr(v, window)
        parts.append(f"{name}最近 {w.size} 个点：均值 {w.mean():.1f}，标准差 {w.std():.1f}")
    return "；".join(parts) + "。"


def series_summary(series: dict, style: str) -> str:
    """What a dashboard would state: totals (or means) over the last 7 / 14 / 28 steps, and recency."""
    step, unit = series.get("step", "步"), series.get("unit", "")
    out = []
    for name, v in list(zip(series["names"], series["values"]))[:3]:
        x = _arr(v)
        w7, w14, p14, w28 = x[-7:], x[-14:], x[-28:-14], x[-28:]
        if style == "counts":
            nz = np.flatnonzero(x > 0)
            since = int(x.size - 1 - nz[-1]) if nz.size else int(x.size)
            out.append(f"{name}：最近 7 {step}共 {w7.sum():.0f}{unit}，最近 14 {step}共 {w14.sum():.0f}{unit}，"
                       f"再往前 14 {step}共 {p14.sum():.0f}{unit}，最近 28 {step}共 {w28.sum():.0f}{unit}；"
                       f"最近 14 {step}里有 {int((w14 > 0).sum())} {step}有活动，距上一次活动已过 {since} {step}。")
        else:
            out.append(f"{name}：最近 7 {step}均值 {w7.mean():.1f}{unit}，最近 14 {step}均值 {w14.mean():.1f}{unit}，"
                       f"再往前 14 {step}均值 {p14.mean():.1f}{unit}，最近 28 {step}均值 {w28.mean():.1f}{unit}。")
    return "".join(out)


def numbers_long(series: dict, window: int, recent: int = 32, segments: int = 96) -> str:
    """The read window as about 96 segment means, plus the most recent values."""
    out = []
    for name, v in zip(series["names"], series["values"]):
        w = _arr(v, window)
        seg = max(1, math.ceil(w.size / segments))
        means = [w[i: i + seg].mean() for i in range(0, w.size, seg)]
        out.append(f"{name} 全部 {w.size} 个点，每 {seg} 个点的平均值（从早到晚）：" + ", ".join(f"{x:.1f}" for x in means)
                   + f"\n{name} 最近 {min(recent, w.size)} 个值：" + ", ".join(f"{x:.1f}" for x in w[-recent:]))
    return "\n".join(out)


def series_norm(series: dict, keep: int = 256) -> tuple[float, float]:
    """(loc, scale) of the first series' recent history: numeric answers are learnt in these units."""
    v = _arr(series["values"][0], keep)
    return float(v.mean()), float(max(v.std(), 1e-3 * (abs(v.mean()) + 1.0)))


# ----------------------------------------------------------------- the series, as soft tokens
class SeriesEncoder(nn.Module):
    """Frozen Chronos-2 + the trained projector: a series becomes soft tokens in the model's own space."""

    def __init__(self, hidden: int, device, chronos: str, ctx_patches: int, fut_patches: int = 2,
                 max_variates: int = 4):
        super().__init__()
        from chronos import Chronos2Pipeline

        self.pipe = Chronos2Pipeline.from_pretrained(chronos, device_map=device)
        self.chronos = self.pipe.model.eval()
        for p in self.chronos.parameters():
            p.requires_grad_(False)
        d = self.chronos.config.d_model
        self.ctx_patches, self.fut_patches = ctx_patches, fut_patches
        self.patch = self.chronos.chronos_config.input_patch_size
        self.role = nn.Embedding(3, d)
        self.variate = nn.Embedding(max_variates, d)
        self.proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.log_scale = nn.Parameter(torch.zeros(()))
        self.to(device)

    @torch.no_grad()
    def states(self, series: dict) -> tuple[torch.Tensor, list[int]]:
        """Chronos-2 states for the target variates, and each token's role (context / summary / future)."""
        vals = np.asarray(series["values"], dtype=np.float32)
        V, T = vals.shape
        keep = self.ctx_patches * self.patch
        rows, fut_rows = [vals[:, -keep:]], [np.full((V, self.fut_patches * self.patch), np.nan, dtype=np.float32)]
        for name, past in (series.get("past_covariates") or {}).items():
            future = (series.get("future_covariates") or {}).get(name)
            rows.append(np.asarray(past, dtype=np.float32)[None, -keep:])
            f = np.zeros(self.fut_patches * self.patch, dtype=np.float32)
            if future is not None:
                f[: min(len(future), f.size)] = np.asarray(future, dtype=np.float32)[: f.size]
            fut_rows.append(f[None])
        ctx = torch.from_numpy(np.concatenate(rows)).to(self.chronos.device)
        fut = torch.from_numpy(np.concatenate(fut_rows)).to(self.chronos.device)
        group = torch.zeros(ctx.shape[0], dtype=torch.long, device=ctx.device)
        enc, _, _, n_ctx = self.chronos.encode(context=ctx, group_ids=group, future_covariates=fut,
                                               num_output_patches=self.fut_patches)
        h = enc.last_hidden_state if hasattr(enc, "last_hidden_state") else enc[0]
        return h[:V], [0] * n_ctx + [1] + [2] * self.fut_patches

    @torch.no_grad()
    def forward(self, series: dict) -> torch.Tensor:
        h, roles = self.states(series)
        V, L, d = h.shape
        role = self.role(torch.tensor(roles, device=h.device))[None].expand(V, L, d)
        var = self.variate(torch.arange(V, device=h.device))[:, None].expand(V, L, d)
        x = self.proj((h.float() + role + var).reshape(V * L, d))
        return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6) * self.log_scale.exp()

    def n_tokens(self, series: dict) -> int:
        V, T = np.asarray(series["values"]).shape
        return V * (math.ceil(min(T, self.ctx_patches * self.patch) / self.patch) + 1 + self.fut_patches)

    @torch.no_grad()
    def chronos_quantiles(self, series: dict, horizon: int, step: int, window: int) -> np.ndarray:
        """Chronos-2's own 10 / 50 / 90% forecast at `step`, from the same window and covariates."""
        item = {"target": np.asarray(series["values"][0][-window:], dtype=np.float32)}
        past, future = series.get("past_covariates") or {}, series.get("future_covariates") or {}
        if past:
            item["past_covariates"] = {k: np.asarray(v[-window:], dtype=np.float32) for k, v in past.items()}
            item["future_covariates"] = {k: np.asarray((list(future.get(k, [])) + [0.0] * horizon)[:horizon],
                                                       dtype=np.float32) for k in past}
        q, _ = self.pipe.predict_quantiles([item], prediction_length=horizon, quantile_levels=list(LEVELS))
        q = q[0] if isinstance(q, (list, tuple)) else q
        return q.reshape(-1, horizon, 3)[0].float().cpu().numpy()[step]


class QuantityHead(nn.Module):
    """The final state -> three monotone quantiles, in units of the series' own recent scale."""

    def __init__(self, hidden: int, device):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 512), nn.GELU(), nn.Linear(512, 3))
        self.to(device)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        o = self.net(h.float())
        mid = o[..., 1]
        return torch.stack([mid - nn.functional.softplus(o[..., 0]), mid, mid + nn.functional.softplus(o[..., 2])], -1)


class ResidualQuantityHead(QuantityHead):
    """Chronos-2's own quantiles, corrected by what the language model read in the text and context."""

    def forward(self, h: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        o = self.net(h.float())
        mid = base[:, 1] + o[:, 1]
        lo_gap = (base[:, 1] - base[:, 0]).clamp_min(1e-3) * torch.exp(o[:, 0].clamp(-4, 4))
        hi_gap = (base[:, 2] - base[:, 1]).clamp_min(1e-3) * torch.exp(o[:, 2].clamp(-4, 4))
        return torch.stack([mid - lo_gap, mid, mid + hi_gap], -1)


class _QuantityPlan:
    qtype = "quantity"

    def __init__(self, block: str):
        self.block, self.primer, self.token_ids, self.keys = block, sf.ANSWER_PRIMER, [], []


class SenseT:
    """The released Sense-T model: decisions and forecasts over a series, with no text generated."""

    def __init__(self, path: str, device: str = "cuda", dtype=torch.bfloat16, chronos: str | None = None):
        import transformers

        self.path = str(path)
        cfg = json.loads((Path(path) / "sense_t.json").read_text(encoding="utf-8"))
        if cfg.get("format") != sf.FORMAT_ID:
            raise ValueError(f"model expects prompt format {cfg.get('format')!r}; this file implements {sf.FORMAT_ID!r}")
        self.cfg = cfg
        self.mode = cfg.get("mode", "mixed")
        self.window = int(cfg.get("window", 256))
        self.stats_window = int(cfg.get("stats_window", 0)) or self.window
        self.summary = cfg.get("summary", "off")
        self.temperature: dict[str, float] = cfg.get("temperature", {})
        self.tok = transformers.AutoTokenizer.from_pretrained(self.path)
        self.model = transformers.AutoModelForImageTextToText.from_pretrained(self.path, dtype=dtype,
                                                                              device_map=device).eval()
        self.device = next(self.model.parameters()).device
        self.pad_id = self.model.config.text_config.pad_token_id
        hidden = self.model.config.text_config.hidden_size
        local = Path(path) / "chronos-2"
        self.encoder = SeriesEncoder(hidden, self.device, chronos or (str(local) if local.exists() else "amazon/chronos-2"),
                                     ctx_patches=self.window // 16)
        self.encoder.load_state_dict(torch.load(Path(path) / "projector.pt", map_location=self.device), strict=False)
        self.head = None
        if (Path(path) / "qhead.pt").exists():
            Head = ResidualQuantityHead if cfg.get("qty_head") == "residual" else QuantityHead
            self.head = Head(hidden, self.device)
            self.head.load_state_dict(torch.load(Path(path) / "qhead.pt", map_location=self.device))
        self._text = next(m for m in self.model.modules()
                          if hasattr(m, "get_per_layer_inputs") and hasattr(m, "embed_tokens"))

    # ------------------------------------------------------------- prompt
    def state_text(self, state: str, series: dict, context: str = "") -> str:
        base = "\n\n".join(x for x in (state or "", context or "") if x)
        base = f"{base}\n{window_stats(series, self.stats_window)}"
        if self.summary != "off":
            base = f"{base}\n{series_summary(series, self.summary)}"
        digits = f"{numbers_long(series, self.window)}\n" if self.mode == "mixed" else ""
        return f"{base}\n{digits}{SERIES_MARK}\n"

    def _plan(self, st: str, question: dict):
        if question["type"] == "quantity":
            return _QuantityPlan(f"Question: {question['instructions']}\nReply with a number.\n")
        return sf.plan(self.tok, st, question)

    def _inputs(self, state: str, series: dict, question: dict, context: str = ""):
        st = self.state_text(state, series, context)
        plan = self._plan(st, question)
        prompt = sf._chat(self.tok, sf.messages(st, plan.block), tokenize=False) + plan.primer
        left, right = prompt.split(SERIES_MARK)
        a = self.tok(left, add_special_tokens=False).input_ids
        b = self.tok(right, add_special_tokens=False).input_ids
        n = self.encoder.n_tokens(series)
        ids = torch.tensor([a + [self.pad_id] * n + b], device=self.device)
        emb = self.model.get_input_embeddings()(ids)
        ple = self._text.get_per_layer_inputs(ids, emb)
        soft = self.encoder(series).to(emb.dtype)
        if soft.shape[0] != n:
            raise RuntimeError(f"series produced {soft.shape[0]} tokens for {n} slots")
        emb = torch.cat([emb[0, : len(a)], soft, emb[0, len(a) + n:]], dim=0)[None]
        att = torch.ones(ids.shape, dtype=torch.long, device=self.device)
        return emb, ple, att, (att.cumsum(-1) - 1).clamp_min(0), plan

    # ------------------------------------------------------------- answers
    @torch.no_grad()
    def ask(self, question: dict, state: str = "", series: dict | None = None, context: str = "",
            horizon: int = 8) -> dict:
        """One typed question about a series -> {"type", "probabilities", "confidence", ...} or quantiles."""
        if series is None:
            raise ValueError("Sense-T answers questions about a series; pass series=...")
        emb, ple, att, pos, plan = self._inputs(state, series, question, context)
        want_h = plan.qtype == "quantity"
        out = self.model(inputs_embeds=emb, per_layer_inputs=ple, attention_mask=att, position_ids=pos,
                         use_cache=False, logits_to_keep=1, output_hidden_states=want_h)
        if want_h:
            if self.head is None:
                raise RuntimeError("this model has no quantity head")
            h = out.hidden_states[-1][:, -1]
            loc, scale = series_norm(series)
            if isinstance(self.head, ResidualQuantityHead):
                base = (self.encoder.chronos_quantiles(series, horizon, question["step"], self.window) - loc) / scale
                q = self.head(h, torch.tensor(base[None], dtype=torch.float32, device=h.device))
            else:
                q = self.head(h)
            q = q[0].float().cpu().numpy() * scale + loc
            return {"type": "quantity", "step": question["step"], "q10": float(q[0]), "q50": float(q[1]),
                    "q90": float(q[2]), "unit": series.get("unit", "")}
        t = float(self.temperature.get(plan.qtype, self.temperature.get("default", 1.0)))
        p = [float(x) for x in sf.probabilities(out.logits[0, -1], plan, t).tolist()]
        return _answer(plan, p)

    def decide(self, state: str, questions: dict[str, dict], series: dict, horizon: int = 8) -> dict[str, dict]:
        """Several questions about one series; each is answered independently."""
        return {qid: self.ask(q, state, series, horizon=horizon) for qid, q in questions.items()}

    def noul(self, state: str, question: str, series: dict, horizon: int = 8) -> float:
        """P(yes) for a yes/no question about the series."""
        return self.ask({"type": "noul", "instructions": question}, state, series, horizon=horizon)["noul"]

    def choice(self, state: str, question: str, options: dict[str, str], series: dict, open_set: bool = False,
               horizon: int = 8) -> dict:
        """options: key -> description. open_set adds a "none of these" option (key "__other__")."""
        return self.ask({"type": "choice", "instructions": question, "criteria": options, "open_set": open_set},
                        state, series, horizon=horizon)

    def score(self, state: str, question: str, levels: Sequence[str], series: dict, horizon: int = 8) -> dict:
        """levels: ordered low -> high. Returns the distribution and its expected level."""
        return self.ask({"type": "score", "instructions": question, "criteria": list(levels)}, state, series,
                        horizon=horizon)

    def forecast(self, series: dict, step: int, horizon: int = 8, state: str = "",
                 question: str | None = None) -> dict:
        """The series' value `step` steps ahead as 10 / 50 / 90% quantiles, in the series' own units."""
        name = series["names"][0]
        q = question or f"第 {step + 1} 个{series.get('step', '步')}的{name}大约是多少{series.get('unit', '')}？"
        return self.ask({"type": "quantity", "instructions": q, "step": step, "levels": list(LEVELS)},
                        state, series, horizon=horizon)
