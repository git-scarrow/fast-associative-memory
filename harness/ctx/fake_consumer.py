#!/usr/bin/env python3
"""Deterministic fake consumer for PR-13 harness tests and dry runs.

``is_real`` is False and cannot be set True: every gate in
``harness/ctx/replay.py`` that guards a real render keys on that flag, so
nothing in this module can stand in for the pinned Qwen3-8B consumer.

The fake is a **pipeline exerciser, not a model**. Its answers carry no
scientific content and no run using it may be scored, reported, or
compared to a registered gate. What it does provide is a fixed, seeded
response for every prompt, so the runner's parsing, resume, reconciliation,
and boundary checks are testable without weights.

Modes select the shape of the output the §8.4 parser must cope with:

    wellformed      one JSON object, nothing else
    extra_prose     the object wrapped in chatter (flagged, not failed)
    no_object       prose only (unparseable: no_balanced_json_object)
    extra_field     an extra key (unparseable: wrong_field_set)
    empty_answer    "answer": "" (unparseable: bad_answer)
    bad_hedged      "hedged": "yes" (unparseable: bad_hedged)
    cap_truncated   an object the token cap left unclosed (unparseable)
    mixed           per-prompt rotation through the above, seeded by digest
"""

import hashlib
import json

MODES = ("wellformed", "extra_prose", "no_object", "extra_field",
         "empty_answer", "bad_hedged", "cap_truncated")

# `mixed` weights the malformed shapes down so a dry run still looks like
# a plausible generation stream rather than a parser torture test.
_MIXED_CYCLE = ("wellformed",) * 6 + ("extra_prose", "no_object",
                                      "extra_field", "cap_truncated")


class FakeConsumer:
    """Deterministic, offline, weight-free. Never a real render."""

    pin_id = "fake-consumer-v1"

    def __init__(self, mode="wellformed"):
        if mode != "mixed" and mode not in MODES:
            raise ValueError(f"unknown fake-consumer mode {mode!r}")
        self._mode = mode
        self.calls = 0

    # The gate in replay.assert_render_permitted reads this. It is a
    # read-only property so no test or caller can promote the fake.
    @property
    def is_real(self):
        return False

    @staticmethod
    def count_tokens(text):
        return len(text.split())

    def _digest(self, prompt):
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def _mode_for(self, digest):
        if self._mode != "mixed":
            return self._mode
        return _MIXED_CYCLE[int(digest[:8], 16) % len(_MIXED_CYCLE)]

    def _answer_for(self, prompt, digest):
        """Governance-reactive only so the dry run exercises both branches
        of the typed `hedged` field. It is not a model of anything."""
        if "unresolved (" in prompt or "escalation available" in prompt:
            return "unresolved", True
        if "WITHDRAWN:" in prompt or "[caveat:" in prompt:
            return f"answer-{digest[:8]}", True
        if "[governed context" not in prompt and "Context:" not in prompt:
            return f"answer-{digest[:8]}", True     # `none` arm: no evidence
        return f"answer-{digest[:8]}", False

    def generate(self, prompt, max_new_tokens=256):
        self.calls += 1
        digest = self._digest(prompt)
        mode = self._mode_for(digest)
        answer, hedged = self._answer_for(prompt, digest)
        obj = json.dumps({"answer": answer, "hedged": hedged})

        if mode == "wellformed":
            raw = obj
        elif mode == "extra_prose":
            raw = f"Sure — here is the result:\n{obj}\nHope that helps."
        elif mode == "no_object":
            raw = "I am not able to answer that from the context provided."
        elif mode == "extra_field":
            raw = json.dumps({"answer": answer, "hedged": hedged,
                              "confidence": 0.87})
        elif mode == "empty_answer":
            raw = json.dumps({"answer": "", "hedged": hedged})
        elif mode == "bad_hedged":
            raw = json.dumps({"answer": answer, "hedged": "yes"})
        elif mode == "cap_truncated":
            raw = obj[:-1]                      # the closing brace never arrives
        else:                                   # unreachable; modes validated
            raise AssertionError(mode)

        # Honour the §8.4 output-token limit under the fake's own
        # whitespace tokenizer, so a cap that is too small truncates here
        # exactly as it would in the real consumer.
        tokens = raw.split(" ")
        if len(tokens) > max_new_tokens:
            raw = " ".join(tokens[:max_new_tokens])
        return raw
