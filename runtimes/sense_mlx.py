"""Gizzai Sense on MLX (Apple silicon): the typed readout, on-device.

MLX hands you the logits in Python, so the readout is the whole port: build the released prompt, run one
forward pass, read the option tokens at the last position, softmax. Nothing is generated.

    pip install mlx-lm
    python -m mlx_lm convert --hf-path Gizzai-Sense-E2B --mlx-path Sense-E2B-mlx -q --q-bits 4
    python runtimes/sense_mlx.py --model Sense-E2B-mlx --tokenizer Gizzai-Sense-E2B --demo

Quantising moves probabilities, and Sense's product is the probability, so a build is "supported" only
after `scripts/conformance.py --runtime mlx` has compared it with the reference on the same rows.

**Untested by its author**: this machine has no Apple silicon. The code follows the reference
implementation line for line, and the conformance run is what turns that into a claim.
"""
from __future__ import annotations

import json
import math
import sys
from typing import Any, Sequence

sys.path.insert(0, "/root/jev-local")
try:
    from jevlocal import sense_format as sf
except ImportError:
    import sense_format as sf


def _entropy_confidence(p: Sequence[float]) -> float:
    if len(p) < 2:
        return 1.0
    h = -sum(x * math.log(max(x, 1e-12)) for x in p)
    return max(0.0, min(1.0, 1.0 - h / math.log(len(p))))


def _answer(plan, p: list[float]) -> dict:
    dist = dict(zip(plan.keys, p))
    if plan.qtype == "noul":
        return {"type": "noul", "noul": p[0], "probabilities": dist, "confidence": abs(2 * p[0] - 1)}
    if plan.qtype == "choice":
        best = max(range(len(p)), key=p.__getitem__)
        return {"type": "choice", "choice": plan.keys[best], "probabilities": dist, "confidence": _entropy_confidence(p)}
    return {"type": "score", "score": sum(i * x for i, x in enumerate(p)), "probabilities": dist,
            "confidence": _entropy_confidence(p)}


class SenseMLX:
    """The released readout, running on MLX weights converted from the same checkpoint."""

    def __init__(self, model: str, tokenizer: str | None = None, temperature: dict[str, float] | None = None):
        import mlx.core as mx
        from mlx_lm import load

        self.mx = mx
        self.model, wrapper = load(model)
        self.tok = getattr(wrapper, "_tokenizer", wrapper)       # sense_format needs the HF tokenizer
        if tokenizer:                                            # the release dir, for the exact prompt + temperatures
            from transformers import AutoTokenizer

            self.tok = AutoTokenizer.from_pretrained(tokenizer)
        self.temperature = temperature or {}
        src = tokenizer or model
        try:
            self.temperature = self.temperature or json.loads(
                open(f"{src}/sense_readout.json", encoding="utf-8").read()).get("temperature", {})
        except OSError:
            pass

    def ask(self, question: dict, state: Any = "") -> dict:
        mx = self.mx
        st = sf.render_state(state)
        plan = sf.plan(self.tok, st, question)
        text = sf._chat(self.tok, sf.messages(st, plan.block), tokenize=False) + plan.primer
        ids = self.tok(text, add_special_tokens=False).input_ids   # the template already carries BOS
        logits = self.model(mx.array([ids]))[0, -1].astype(mx.float32)
        sel = mx.take(logits, mx.array(list(plan.token_ids)))
        t = float(self.temperature.get(plan.qtype, self.temperature.get("default", 1.0)))
        p = mx.softmax(sel / t)
        return _answer(plan, [float(x) for x in p.tolist()])

    def noul(self, state: Any, question: str) -> float:
        return self.ask({"type": "noul", "instructions": question}, state)["noul"]

    def choice(self, state: Any, question: str, options: dict[str, str], open_set: bool = False) -> dict:
        return self.ask({"type": "choice", "instructions": question, "criteria": options, "open_set": open_set}, state)

    def score(self, state: Any, question: str, levels: Sequence[str]) -> dict:
        return self.ask({"type": "score", "instructions": question, "criteria": list(levels)}, state)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="the MLX-converted model directory")
    ap.add_argument("--tokenizer", default="", help="the original release directory (prompt + temperatures)")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    s = SenseMLX(a.model, tokenizer=a.tokenizer or None)
    if a.demo:
        print(s.noul("部署连续失败两次，客户已经看到 500 错误。", "需要立即处理吗？"))
        print(s.choice("Checkout returns 502 for every EU customer since 09:00.", "Which team owns this?",
                       {"infra": "Infrastructure", "billing": "Billing"}, open_set=True))
