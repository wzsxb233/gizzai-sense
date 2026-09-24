"""Gizzai Sense on vLLM: the typed readout, served.

vLLM can restrict sampling to a set of token ids (`allowed_token_ids`). Every other token is masked
before the softmax, so the logprobs that come back are exactly the distribution Sense reads: the answer's
own options, renormalised. The prompt is built here by the released `sense_format`, so it is byte-identical
to the reference implementation.

    vllm serve /root/release/Gizzai-Sense-E2B --served-model-name sense --port 8000
    python runtimes/sense_vllm.py --endpoint http://127.0.0.1:8000 --demo

The offline API works the same way:
    SamplingParams(max_tokens=1, logprobs=len(ids), allowed_token_ids=ids)

A build counts as supported only after `scripts/conformance.py` has compared it to the reference.
"""
from __future__ import annotations

import json
import math
import sys
from typing import Any, Sequence

import requests

sys.path.insert(0, "/root/jev-local")
try:
    from jevlocal import sense_format as sf
except ImportError:                       # next to the released files
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


class SenseVLLM:
    """The released readout against a vLLM server holding the same weights."""

    def __init__(self, endpoint: str = "http://127.0.0.1:8000", model: str = "sense",
                 tokenizer: str | None = None, temperature: dict[str, float] | None = None, timeout: float = 120.0):
        from transformers import AutoTokenizer

        self.endpoint, self.model, self.timeout = endpoint.rstrip("/"), model, timeout
        src = tokenizer or "/root/release/Gizzai-Sense-E2B"
        self.tok = AutoTokenizer.from_pretrained(src)
        self.temperature = temperature or {}
        try:
            self.temperature = self.temperature or json.loads(open(f"{src}/sense_readout.json", encoding="utf-8").read()).get("temperature", {})
        except OSError:
            pass

    def ask(self, question: dict, state: Any = "") -> dict:
        st = sf.render_state(state)
        plan = sf.plan(self.tok, st, question)
        prompt = sf._chat(self.tok, sf.messages(st, plan.block), tokenize=False) + plan.primer
        ids = list(plan.token_ids)
        body = {"model": self.model, "prompt": prompt, "max_tokens": 1, "temperature": 1.0,
                "logprobs": min(20, len(ids)), "allowed_token_ids": ids}
        r = requests.post(f"{self.endpoint}/v1/completions", json=body, timeout=self.timeout)
        r.raise_for_status()
        lp = r.json()["choices"][0]["logprobs"]
        top = (lp.get("top_logprobs") or [{}])[0]
        # the server may key by token string; fall back to the decoded ids in order
        probs = []
        by_str = {k: math.exp(v) for k, v in top.items()}
        for i in ids:
            s = self.tok.decode([i])
            probs.append(by_str.get(s, by_str.get(s.strip(), 0.0)))
        if sum(probs) <= 0:
            raise RuntimeError(f"no option probabilities returned; got {list(top)[:5]}")
        t = float(self.temperature.get(plan.qtype, self.temperature.get("default", 1.0)))
        lg = [math.log(max(x, 1e-12)) / t for x in probs]
        m = max(lg)
        e = [math.exp(x - m) for x in lg]
        return _answer(plan, [x / sum(e) for x in e])

    def noul(self, state: Any, question: str) -> float:
        return self.ask({"type": "noul", "instructions": question}, state)["noul"]

    def choice(self, state: Any, question: str, options: dict[str, str], open_set: bool = False) -> dict:
        return self.ask({"type": "choice", "instructions": question, "criteria": options, "open_set": open_set}, state)

    def score(self, state: Any, question: str, levels: Sequence[str]) -> dict:
        return self.ask({"type": "score", "instructions": question, "criteria": list(levels)}, state)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--endpoint", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="sense")
    ap.add_argument("--tokenizer", default="/root/release/Gizzai-Sense-E2B")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    s = SenseVLLM(a.endpoint, model=a.model, tokenizer=a.tokenizer)
    if a.demo:
        print(s.noul("部署连续失败两次，客户已经看到 500 错误。", "需要立即处理吗？"))
        print(s.choice("Checkout returns 502 for every EU customer since 09:00.", "Which team owns this?",
                       {"infra": "Infrastructure", "billing": "Billing"}, open_set=True))
