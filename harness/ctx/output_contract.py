#!/usr/bin/env python3
"""PR-13 consumer output contract — parser (memo §8.4).

Normative artifacts:
    harness/ctx/schema/consumer_output.schema.json
    harness/ctx/policy/consumer_output_contract_v1.json

Registered rules, implemented exactly:
  * the parser extracts the FIRST balanced JSON object in the raw
    output; any non-whitespace characters outside that object set a
    per-row ``extra_prose`` flag (recorded and reported, never a
    failure by itself);
  * no balanced object, an object left unbalanced by the token cap, or
    an object failing schema validation (missing/extra fields, wrong
    types, empty answer) → the row is ``unparseable``;
  * ``unparseable`` scores as wrong for the arm that produced it, and
    feeds a reported malformed-rate per arm × B (scoring side, not
    here).

Deterministic, stdlib-only, no model access.
"""

MAX_NEW_TOKENS = 256
STOPPING = ("eos", "max_new_tokens")


def _first_balanced_object(text):
    """Return (json_substring, start, end) of the first balanced JSON
    object in ``text``, or None. Brace tracking respects string
    literals and escapes."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start:i + 1], start, i + 1
        # This '{' never closes (e.g. cap-truncated output): no balanced
        # object starting here, and none can start inside it either —
        # the row is unparseable.
        return None
    return None


def parse_consumer_output(raw):
    """Parse one raw consumer generation under the §8.4 contract.

    Returns a dict:
      {"status": "ok",  "answer": str, "hedged": bool, "extra_prose": bool}
      {"status": "unparseable", "reason": str}
    """
    import json

    found = _first_balanced_object(raw)
    if found is None:
        return {"status": "unparseable", "reason": "no_balanced_json_object"}
    fragment, start, end = found
    try:
        obj = json.loads(fragment)
    except ValueError:
        return {"status": "unparseable", "reason": "invalid_json"}
    if not isinstance(obj, dict):
        return {"status": "unparseable", "reason": "not_an_object"}
    if set(obj.keys()) != {"answer", "hedged"}:
        return {"status": "unparseable", "reason": "wrong_field_set"}
    if not isinstance(obj["answer"], str) or not obj["answer"]:
        return {"status": "unparseable", "reason": "bad_answer"}
    if not isinstance(obj["hedged"], bool):
        return {"status": "unparseable", "reason": "bad_hedged"}
    extra = (raw[:start] + raw[end:]).strip() != ""
    return {"status": "ok", "answer": obj["answer"], "hedged": obj["hedged"],
            "extra_prose": extra}
