# Copyright 2026 邓颐村 (Deng Yicun). Licensed by GizzAI Technology Limited Co. (广州叽喳科技有限公司)
# under the Gizzai Sense License -- see LICENSE. Commercial and academic collaboration: dengyicun@gizzai.com
"""Gizzai Sense prompt format 2: the one contract between the weights and any runtime.

A Sense model answers a typed question by reading the next-token distribution
at the end of a fixed prompt, restricted to one token per option. Which prompt,
and which tokens, is decided here and nowhere else: the trainer, our servers
and the public readout all import this module, so a model is always asked in
exactly the format it was trained on. It ships with the weights.

    system  SYSTEM_PROMPT
    user    [images / video]  <state>\\n{state}\\n</state>\\n\\n{question block}  [audio]
    model   Answer:{primer}                      <- read the next token here

Gemma 4's chat template trims every text part and joins the parts with nothing
between them, so all text goes in ONE part and media sit on either side of it,
in the order the Gemma 4 card recommends (images before text, audio after).
A text-only prompt renders byte-identically to format 1.

Depends on nothing but torch; question objects may be plain dicts or any object
with .type / .instructions / .criteria.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence

import torch

FORMAT_ID = "gizzai-sense/2"
SYSTEM_PROMPT = (
    "You are a decision engine. You read a STATE and answer one question about "
    "it by emitting exactly one label. Never explain, never add words."
)
ANSWER_PRIMER = "Answer:"
# Gemma tokenises a space as its own token: ending the primer with the space
# makes each label the next token. Tried in order; first distinct tokenisation wins.
PRIMER_VARIANTS: tuple[tuple[str, str], ...] = ((" ", ""), ("", " "))
LETTERS = [chr(ord("A") + i) for i in range(26)]
OTHER = "__other__"
OTHER_DESCRIPTION = "None of the listed options fits"
VISUAL_KINDS = ("image", "video")
AUDIO_KINDS = ("audio",)


def _get(q: Any, key: str, default: Any = None) -> Any:
    return q.get(key, default) if isinstance(q, dict) else getattr(q, key, default)


def render_state(state: Any) -> str:
    if state is None:
        return ""
    if isinstance(state, str):
        return state
    if isinstance(state, (list, tuple)) and all(isinstance(x, str) for x in state):
        return "\n".join(state)
    return json.dumps(state, ensure_ascii=False, indent=2)


def question_block(question: Any) -> tuple[str, list[str], list[str]]:
    """(text, labels, keys): what the model reads, what it may answer, what callers get back."""
    qtype, instr = _get(question, "type"), _get(question, "instructions")
    if qtype == "noul":
        return f"Question: {instr}\nReply with yes or no.\n", ["yes", "no"], ["yes", "no"]
    if qtype == "choice":
        criteria = dict(_get(question, "criteria"))
        if _get(question, "open_set", False) and OTHER not in criteria:
            criteria[OTHER] = OTHER_DESCRIPTION
        keys = list(criteria)
        if not 2 <= len(keys) <= 26:
            raise ValueError("choice needs 2-26 options")
        labels = LETTERS[: len(keys)]
        lines = "\n".join(f"{lab}) {criteria[k]}" for lab, k in zip(labels, keys))
        text = (f"Question: {instr}\nOptions:\n{lines}\n"
                "Reply with the single letter of the best option.\n")
        return text, labels, keys
    if qtype == "score":
        keys = list(_get(question, "criteria"))
        if not 2 <= len(keys) <= 10:
            raise ValueError("score needs 2-10 levels")
        labels = [str(i) for i in range(len(keys))]
        lines = "\n".join(f"{i} = {lvl}" for i, lvl in enumerate(keys))
        text = (f"Question: {instr}\nLevels:\n{lines}\n"
                "Reply with the single digit of the level that fits best.\n")
        return text, labels, keys
    raise ValueError(f"unknown question type {qtype!r}")


def body(state_text: str, block: str) -> str:
    return f"<state>\n{state_text}\n</state>\n\n{block}"


def messages(state_text: str, block: str, media: Sequence[dict] = ()) -> list[dict]:
    """Chat messages. media items look like {"type": "image", "image": ...}."""
    head = [{"role": "system", "content": SYSTEM_PROMPT}]
    text = body(state_text, block)
    if not media:
        return head + [{"role": "user", "content": text}]
    for m in media:
        if m.get("type") not in VISUAL_KINDS + AUDIO_KINDS:
            raise ValueError(f"unsupported media type {m.get('type')!r}")
    content = [m for m in media if m["type"] in VISUAL_KINDS]
    content.append({"type": "text", "text": text})
    content += [m for m in media if m["type"] in AUDIO_KINDS]
    return head + [{"role": "user", "content": content}]


def _chat(renderer, msgs, **kw):
    try:
        return renderer.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, **kw)
    except TypeError:  # templates without a thinking switch
        return renderer.apply_chat_template(msgs, add_generation_prompt=True, **kw)


@dataclass
class Plan:
    qtype: str
    keys: list[str]
    labels: list[str]
    token_ids: list[int]
    primer: str  # ANSWER_PRIMER + the variant's suffix, appended after the chat template
    block: str


def _first_new_token(tok, prompt: str, continuation: str) -> int:
    a = tok(prompt, add_special_tokens=False).input_ids
    b = tok(prompt + continuation, add_special_tokens=False).input_ids
    i = 0
    while i < len(a) and i < len(b) and a[i] == b[i]:
        i += 1
    if i >= len(b) or i != len(a):
        raise ValueError(f"{continuation!r} does not start a clean new token")
    return b[i]


def plan(tok, state_text: str, question: Any) -> Plan:
    """Resolve each label to the single token the model would emit for it.

    Resolved against the real rendered text prompt, so a tokeniser merge at the
    primer boundary cannot silently shift which token is scored. Media do not
    change the tail of the prompt, so the same ids serve media prompts.
    """
    block, labels, keys = question_block(question)
    base = _chat(tok, messages(state_text, block), tokenize=False) + ANSWER_PRIMER
    problem = "no variant"
    for primer_suffix, label_prefix in PRIMER_VARIANTS:
        try:
            ids = [_first_new_token(tok, base + primer_suffix, label_prefix + lab) for lab in labels]
        except ValueError as exc:
            problem = str(exc)
            continue
        if len(set(ids)) == len(ids):
            return Plan(_get(question, "type"), keys, labels, ids, ANSWER_PRIMER + primer_suffix, block)
        problem = f"labels collide: {labels} -> {ids}"
    raise ValueError(f"no usable label tokenisation ({problem})")


def encode(tok, processor, state: Any, question: Any, media: Sequence[dict] = ()):
    """Model inputs for one question, ending at the primer, plus its Plan."""
    state_text = render_state(state)
    p = plan(tok, state_text, question)
    return encode_block(tok, processor, state_text, p.block, p.primer, media), p


def encode_block(tok, processor, state_text: str, block: str, primer: str, media: Sequence[dict] = ()):
    """Tokenised prompt for an already-planned question, ending at the primer.

    Text-only prompts go through the tokenizer, media prompts through the
    processor (which expands media placeholders); the primer is appended as
    plain text tokens in both cases.
    """
    primer_ids = tok(primer, add_special_tokens=False).input_ids
    if not media:
        prompt = _chat(tok, messages(state_text, block), tokenize=False) + primer
        return dict(tok(prompt, return_tensors="pt", add_special_tokens=False))
    if processor is None:
        raise ValueError("media inputs need the processor")
    enc = _chat(processor, messages(state_text, block, media),
                tokenize=True, return_dict=True, return_tensors="pt")
    n = len(primer_ids)
    seq = enc["input_ids"].shape[1]
    batch = {}
    for key, value in enc.items():
        if hasattr(value, "shape") and value.ndim == 2 and value.shape[1] == seq:
            if key == "input_ids":
                tail = torch.tensor([primer_ids], dtype=value.dtype)
            elif key == "attention_mask":
                tail = torch.ones((1, n), dtype=value.dtype)
            else:  # token-type style masks: the primer is plain text
                tail = torch.zeros((1, n), dtype=value.dtype)
            batch[key] = torch.cat([value, tail], dim=1)
        else:
            batch[key] = value
    return batch


def probabilities(last_logits: torch.Tensor, p: Plan, temperature: float = 1.0) -> torch.Tensor:
    """Restricted softmax over the plan's label tokens, temperature-scaled."""
    ids = torch.tensor(p.token_ids, device=last_logits.device)
    return torch.softmax(last_logits.float()[ids] / temperature, dim=-1)
