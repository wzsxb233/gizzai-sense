# Copyright 2026 邓颐村 (Deng Yicun). Licensed by GizzAI Technology Limited Co. (广州叽喳科技有限公司)
# under the Gizzai Sense License -- see LICENSE. Commercial and academic collaboration: dengyicun@gizzai.com
"""Gizzai Sense (叽喳 Sense): typed decisions with calibrated probabilities.

    from sense import Sense

    s = Sense("Gizzai-Sense-E4B")
    s.noul("部署连续失败两次，客户已经看到 500 错误。", "需要立即处理吗？")    # P(yes)
    s.choice(ticket, "Which team owns this?", {"infra": "Infrastructure", "billing": "Billing"})
    s.score(review, "How satisfied is the customer?", ["angry", "unhappy", "neutral", "happy"])
    s.noul("", "Is there a dog in the picture?", media=[{"type": "image", "image": "photo.jpg"}])

Every answer is one forward pass: the model reads the prompt and we take the
next-token distribution restricted to one token per option, so the output is a
probability distribution over exactly the options you gave -- no text to parse,
nothing outside the set. Probabilities are temperature-calibrated with the
values shipped in sense_readout.json.

This is the reference implementation: one question per forward pass, no
batching or cache sharing, media within the base model's native limits (audio
up to 30 s per clip). It reproduces the published evaluation numbers.
Optimised serving (shared-prefix batching, long audio and video, multi-step
decision graphs, open-set generation), the training and RL frameworks, and
commercial or academic collaboration: dengyicun@gizzai.com.

Not for fully automated decisions with legal or similarly significant effects
on people; keep a human in the loop for those (see LICENSE and USE_POLICY).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch

try:
    from . import sense_format as sf
except ImportError:  # used as a plain file next to sense_format.py
    import sense_format as sf

AUDIO_SR = 16_000
MAX_AUDIO_S = 30.0  # Gemma 4 truncates longer clips without saying so


def _load_model(path: str, dtype, device):
    import transformers

    errors = []
    for name in ("AutoModelForImageTextToText", "AutoModelForCausalLM"):
        cls = getattr(transformers, name, None)
        if cls is None:
            continue
        for kw in ("dtype", "torch_dtype"):
            try:
                return cls.from_pretrained(path, **{kw: dtype}, device_map=device)
            except TypeError as exc:
                errors.append(f"{name}/{kw}: {exc}")
            except (ValueError, KeyError, OSError) as exc:
                errors.append(f"{name}: {exc}")
                break
    raise RuntimeError("could not load the model:\n" + "\n".join(errors))


def _confidence(p: Sequence[float]) -> float:
    """1 - normalised entropy: 0 when uniform, 1 when one option is certain."""
    if len(p) < 2:
        return 1.0
    h = -sum(x * math.log(max(x, 1e-12)) for x in p)
    return max(0.0, min(1.0, 1.0 - h / math.log(len(p))))


def _answer(plan: sf.Plan, p: list[float]) -> dict:
    dist = dict(zip(plan.keys, p))
    if plan.qtype == "noul":
        return {"type": "noul", "noul": p[0], "probabilities": dist, "confidence": abs(2 * p[0] - 1)}
    if plan.qtype == "choice":
        best = max(range(len(p)), key=p.__getitem__)
        return {"type": "choice", "choice": plan.keys[best], "probabilities": dist,
                "confidence": _confidence(p)}
    return {"type": "score", "score": sum(i * x for i, x in enumerate(p)), "probabilities": dist,
            "confidence": _confidence(p)}


class Sense:
    def __init__(self, path: str, device: str = "cuda", dtype=torch.bfloat16, check_format: bool = True):
        from transformers import AutoTokenizer

        self.path = str(path)
        cfg = json.loads((Path(path) / "sense_readout.json").read_text(encoding="utf-8"))
        if check_format and cfg.get("format") != sf.FORMAT_ID:
            raise ValueError(f"model expects prompt format {cfg.get('format')!r}; "
                             f"this readout implements {sf.FORMAT_ID!r}")
        self.name = cfg.get("name", Path(path).name)
        self.temperature: dict[str, float] = cfg.get("temperature", {})
        self.tok = AutoTokenizer.from_pretrained(self.path)
        self._processor = None
        self.model = _load_model(self.path, dtype, device)
        self.model.eval()
        self.device = next(self.model.parameters()).device

    @property
    def processor(self):
        if self._processor is None:
            from transformers import AutoProcessor

            self._processor = AutoProcessor.from_pretrained(self.path)
        return self._processor

    def _media(self, media: Sequence[dict]) -> list[dict]:
        out = []
        for m in media:
            m = dict(m)
            if m.get("type") == "audio":
                a = m["audio"]
                if isinstance(a, (str, Path)):
                    from transformers.audio_utils import load_audio

                    a = load_audio(str(a), sampling_rate=AUDIO_SR)
                if len(a) / AUDIO_SR > MAX_AUDIO_S + 1e-6:
                    raise ValueError(f"audio clip is {len(a) / AUDIO_SR:.1f} s; the model reads at most "
                                     f"{MAX_AUDIO_S:.0f} s per clip. Split it, or contact us for long-media serving.")
                m["audio"] = a
            out.append(m)
        return out

    def _temperature(self, qtype: str) -> float:
        return float(self.temperature.get(qtype, self.temperature.get("default", 1.0)))

    @torch.no_grad()
    def ask(self, question: dict, state: Any = "", media: Sequence[dict] = ()) -> dict:
        """One typed question -> {"type", "probabilities", "confidence", and noul|choice|score}."""
        media = self._media(media)
        batch, plan = sf.encode(self.tok, self.processor if media else None, state, question, media)
        batch = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in batch.items()}
        logits = self.model(**batch, use_cache=False, logits_to_keep=1).logits[0, -1]
        p = [float(x) for x in sf.probabilities(logits, plan, self._temperature(plan.qtype)).tolist()]
        return _answer(plan, p)

    def decide(self, state: Any, questions: dict[str, dict], media: Sequence[dict] = ()) -> dict[str, dict]:
        """Several questions about one state; each is answered independently."""
        return {qid: self.ask(q, state, media) for qid, q in questions.items()}

    def noul(self, state: Any, question: str, media: Sequence[dict] = ()) -> float:
        """P(yes)."""
        return self.ask({"type": "noul", "instructions": question}, state, media)["noul"]

    def choice(self, state: Any, question: str, options: dict[str, str], open_set: bool = False,
               media: Sequence[dict] = ()) -> dict:
        """options: key -> description. open_set adds a "none of these" option (key "__other__")."""
        q = {"type": "choice", "instructions": question, "criteria": options, "open_set": open_set}
        return self.ask(q, state, media)

    def score(self, state: Any, question: str, levels: Sequence[str], media: Sequence[dict] = ()) -> dict:
        """levels: ordered low -> high. Returns the distribution and its expected level."""
        return self.ask({"type": "score", "instructions": question, "criteria": list(levels)}, state, media)
