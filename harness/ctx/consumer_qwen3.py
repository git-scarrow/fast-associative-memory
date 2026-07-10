#!/usr/bin/env python3
"""PR-13 real consumer: the pinned Qwen3-8B, non-thinking (memo §8.1).

Nothing here is invoked by the test suite, and constructing this class is
the moment the scoring run becomes real. Three locks, in order:

1. ``authorize`` must be the exact registered token. There is no default.
2. Every artifact in ``consumer_pin.json`` must be present in the seal
   directory and match its pinned sha256 — checked **before** torch or
   transformers is imported, so an unsealed tree cannot even load a
   framework, let alone weights.
3. Decoding is read from the pin, not from arguments: greedy, temperature
   0, ``max_new_tokens`` from the §8.4 contract, ``enable_thinking=False``
   applied through the pinned chat template.

``is_real`` is True and read-only. ``harness/ctx/replay.py`` refuses to
run a real consumer without a sealed query manifest and a sealed scoring
manifest that pins it.
"""

import json
import os

from harness.ctx.output_contract import MAX_NEW_TOKENS
from harness.ctx.seal_consumer import SEAL_DIR, _expected, _pin, _sha_file

AUTHORIZATION_TOKEN = "pr13-scoring-run"


def verify_seal(seal_dir=None, pin=None):
    """Every pinned artifact present and byte-exact. Returns the pin."""
    pin = pin or _pin()
    seal_dir = seal_dir or SEAL_DIR
    bad, missing = [], []
    for name, want in sorted(_expected(pin).items()):
        path = os.path.join(seal_dir, name)
        if not os.path.exists(path):
            missing.append(name)
        elif _sha_file(path) != want:
            bad.append(name)
    if missing or bad:
        raise RuntimeError(
            f"consumer seal not satisfied — missing {missing}, mismatched "
            f"{bad}; the §8.1 kill forbids rendering against unverified "
            f"artifacts")
    return pin


class Qwen3Consumer:
    """The experimental subject. Never a certified reader (memo §10)."""

    is_real = True

    def __init__(self, authorize, seal_dir=None):
        if authorize != AUTHORIZATION_TOKEN:
            raise PermissionError(
                "the scoring run is separately authorized; construct this "
                f"consumer only with authorize={AUTHORIZATION_TOKEN!r}")
        seal_dir = seal_dir or SEAL_DIR
        pin = verify_seal(seal_dir)

        # Imported only after the seal verifies.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.pin = pin
        self.pin_id = pin["pin_id"]
        self.revision = pin["artifact"]["revision"]
        self._tokenizer = AutoTokenizer.from_pretrained(seal_dir)
        self._model = AutoModelForCausalLM.from_pretrained(
            seal_dir, dtype=torch.bfloat16, device_map="auto")
        self._model.eval()
        self._torch = torch

        decoding = pin["runtime"]["decoding"]
        if decoding["do_sample"] or decoding["max_new_tokens"] != MAX_NEW_TOKENS:
            raise RuntimeError("pin decoding diverges from the §8.4 contract")
        if decoding["enable_thinking"]:
            raise RuntimeError("the family-level pin fixes non-thinking mode")

    def count_tokens(self, text):
        return len(self._tokenizer(text, add_special_tokens=False)["input_ids"])

    def generate(self, prompt, max_new_tokens=MAX_NEW_TOKENS):
        text = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        with self._torch.inference_mode():
            out = self._model.generate(
                **inputs, do_sample=False, temperature=None, top_p=None,
                top_k=None, max_new_tokens=max_new_tokens,
                pad_token_id=self._tokenizer.eos_token_id)
        generated = out[0][inputs["input_ids"].shape[-1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True)

    def runtime_versions(self):
        """Recorded verbatim into the scoring manifest (§5 of the run
        deliverable): what actually produced the numbers."""
        import platform
        import tokenizers
        import torch
        import transformers
        return {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "tokenizers": tokenizers.__version__,
            "device": str(next(self._model.parameters()).device),
            "dtype": str(next(self._model.parameters()).dtype),
        }
