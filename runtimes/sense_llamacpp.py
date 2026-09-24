"""Gizzai Sense on llama.cpp / Ollama: the same typed readout, over a GGUF.

Sense answers by reading the probabilities of the answer's own option tokens at one position -- it never
generates prose. llama.cpp can do exactly that: a grammar restricts the next token to the option set, and
`n_probs` returns the probabilities over what survives. This client builds the released prompt with
`sense_format` (so the text is byte-identical to the reference) and returns the same answer dict as
`sense.py`.

    llama-server -m Gizzai-Sense-E2B-Q8_0.gguf --port 8080 --no-warmup
    python runtimes/sense_llamacpp.py --endpoint http://127.0.0.1:8080 --demo

Quantisation moves probabilities. Sense's product IS the probability, so a build is only "supported"
once `scripts/conformance.py` has measured it against the reference: per-row probability difference,
accuracy, and calibration error.
"""
from __future__ import annotations

import json
import math
import sys
from typing import Any, Sequence

import requests

sys.path.insert(0, "/root/jev-local/release")
sys.path.insert(0, "/root/jev-local/jevlocal")
try:
    from jevlocal import sense_format as sf
except ImportError:
    import sense_format as sf


def _answer(plan, p: list[float]) -> dict:
    dist = dict(zip(plan.keys, p))
    if plan.qtype == "noul":
        return {"type": "noul", "noul": p[0], "probabilities": dist, "confidence": abs(2 * p[0] - 1)}
    if plan.qtype == "choice":
        best = max(range(len(p)), key=p.__getitem__)
        return {"type": "choice", "choice": plan.keys[best], "probabilities": dist,
                "confidence": _entropy_confidence(p)}
    return {"type": "score", "score": sum(i * x for i, x in enumerate(p)), "probabilities": dist,
            "confidence": _entropy_confidence(p)}


def _entropy_confidence(p: Sequence[float]) -> float:
    import math

    if len(p) < 2:
        return 1.0
    h = -sum(x * math.log(max(x, 1e-12)) for x in p)
    return max(0.0, min(1.0, 1.0 - h / math.log(len(p))))


class SenseLlamaCpp:
    """The released readout against a llama.cpp server holding a GGUF of the same weights."""

    pruned_options = 0      # options the server did not report (counted as 0 probability)
    pruned_rows = 0

    def __init__(self, endpoint: str = "http://127.0.0.1:8080", tokenizer: str | None = None,
                 temperature: dict[str, float] | None = None, timeout: float = 120.0, n_probs: int = 64):
        self.endpoint, self.timeout, self.n_probs = endpoint.rstrip("/"), timeout, n_probs
        self.temperature = temperature or {}
        self.tok = None
        if tokenizer:                      # the reference tokenizer, for building the prompt and the plan
            from transformers import AutoTokenizer

            self.tok = AutoTokenizer.from_pretrained(tokenizer)
            cfg = {}
            try:
                cfg = json.loads(open(f"{tokenizer}/sense_readout.json", encoding="utf-8").read())
            except OSError:
                pass
            self.temperature = self.temperature or cfg.get("temperature", {})

    # ------------------------------------------------------------------ prompt
    def _prompt_and_plan(self, state: Any, question: dict):
        """Token ids, not text: the chat template already carries BOS, and a server that adds its own
        would answer a different prompt (seen: choice probabilities off by 0.17 before this)."""
        if self.tok is None:
            raise RuntimeError("pass tokenizer=<release dir> so the prompt matches the reference exactly")
        st = sf.render_state(state)
        plan = sf.plan(self.tok, st, question)
        text = sf._chat(self.tok, sf.messages(st, plan.block), tokenize=False) + plan.primer
        return self.tok(text, add_special_tokens=False).input_ids, plan

    @staticmethod
    def _grammar(options: Sequence[str]) -> str:
        """Only the option strings may be produced, so the reported probabilities cover exactly them."""
        alts = " | ".join(json.dumps(o, ensure_ascii=False) for o in options)
        return f"root ::= {alts}"

    # ------------------------------------------------------------------ answers
    def ask(self, question: dict, state: Any = "") -> dict:
        """One forward pass, then the restricted softmax over the plan's own token ids.

        No grammar: the server's grammar filter works on strings, and the reference scores the token
        " A", not the string "A" -- scoring the wrong tokens moved a three-way choice by 0.026. Instead
        this asks for the pre-sampling logprobs of the top candidates and matches them by token id, which
        is exactly what the reference implementation reads off the logits.
        """
        prompt, plan = self._prompt_and_plan(state, question)
        body = {"prompt": prompt, "n_predict": 1, "n_probs": self.n_probs, "post_sampling_probs": False,
                "temperature": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0, "cache_prompt": True}
        r = requests.post(f"{self.endpoint}/completion", json=body, timeout=self.timeout)
        r.raise_for_status()
        top = ((r.json().get("completion_probabilities") or [{}])[0]).get("top_logprobs") or []
        by_id = {int(it["id"]): float(it["logprob"]) for it in top if "id" in it and "logprob" in it}
        missing = [i for i in plan.token_ids if i not in by_id]
        if missing:
            SenseLlamaCpp.pruned_options += len(missing)
            SenseLlamaCpp.pruned_rows += 1
        if len(missing) == len(plan.token_ids):
            raise RuntimeError(f"none of the option tokens were in the top {self.n_probs}; raise n_probs")
        t = float(self.temperature.get(plan.qtype, self.temperature.get("default", 1.0)))
        # a logprob is the logit minus the same log-partition for every token, so it cancels here
        lg = [by_id.get(i, -60.0) / t for i in plan.token_ids]
        m = max(lg)
        e = [math.exp(x - m) for x in lg]
        s = sum(e)
        return _answer(plan, [x / s for x in e])

    @staticmethod
    def _probs_from(data: dict, opts: Sequence[str]) -> list[float]:
        """The option probabilities from a completion response, renormalised over the options."""
        top = (data.get("completion_probabilities") or [{}])[0]
        items = top.get("probs") or top.get("top_logprobs") or []
        found = {}
        for it in items:
            tokv = it.get("tok_str", it.get("token", ""))
            p = it.get("prob")
            if p is None and "logprob" in it:
                import math

                p = math.exp(it["logprob"])
            for o in opts:
                if tokv == o and o not in found:      # exact token string: " A" is not "A"
                    found[o] = float(p)
        # llama.cpp drops candidates whose probability rounds to zero after the grammar filter. Those
        # options are worth ~0 in the reference too, so they count as 0 -- but the client keeps score,
        # because "an option never came back" would otherwise be invisible in a conformance report.
        missing = [o for o in opts if o not in found]
        if missing:
            SenseLlamaCpp.pruned_options += len(missing)
            SenseLlamaCpp.pruned_rows += 1
        if len(missing) == len(opts):
            raise RuntimeError(f"server reported no option probabilities at all (options {list(opts)})")
        s = sum(found.get(o, 0.0) for o in opts) or 1.0
        return [found.get(o, 0.0) / s for o in opts]

    def noul(self, state: Any, question: str) -> float:
        return self.ask({"type": "noul", "instructions": question}, state)["noul"]

    def choice(self, state: Any, question: str, options: dict[str, str], open_set: bool = False) -> dict:
        return self.ask({"type": "choice", "instructions": question, "criteria": options, "open_set": open_set}, state)

    def score(self, state: Any, question: str, levels: Sequence[str]) -> dict:
        return self.ask({"type": "score", "instructions": question, "criteria": list(levels)}, state)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--endpoint", default="http://127.0.0.1:8080")
    ap.add_argument("--tokenizer", default="/root/release/Gizzai-Sense-E2B")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    s = SenseLlamaCpp(a.endpoint, tokenizer=a.tokenizer)
    if a.demo:
        print(s.noul("部署连续失败两次，客户已经看到 500 错误。", "需要立即处理吗？"))
        print(s.choice("Checkout returns 502 for every EU customer since 09:00.", "Which team owns this?",
                       {"infra": "Infrastructure", "billing": "Billing"}, open_set=True))
