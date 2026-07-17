"""Canonical input seals for reproducible five-arm runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import torch

from . import ARM_NAMES, CONTEXT_BUDGET_TOKENS
from . import scoring as _scoring_module
from .context import POLICY_VERSION
from .models import MemoryQuestion, MemoryRecord
from .preregistration import validate_registration
from .retrievers import TensorLike
from .scoring import SCORING_VERSION

__all__ = [
    "MANIFEST_VERSION",
    "SCORING_VERSION",
    "EVIDENCE_CLASSES",
    "canonical_json",
    "policy_sha256",
    "scoring_module_sha256",
    "seal_manifest",
    "load_manifest",
    "verify_manifest",
]

MANIFEST_VERSION = "memory-eval-manifest-v2"

#: A seal must say what it is. ``plumbing`` cannot carry a registration and is
#: never admissible as evidence; ``scoring-run`` must carry a complete one
#: (gate G-I1) and binds the treatment (gate G-I2).
EVIDENCE_CLASSES = ("plumbing", "scoring-run")


def input_fingerprints(
    records: Sequence[MemoryRecord],
    questions: Sequence[MemoryQuestion],
    record_embeddings: Mapping[str, TensorLike],
    query_embeddings: Mapping[str, TensorLike],
) -> dict[str, str]:
    """Fingerprint the sealed INPUTS only, independent of the treatment.

    Separating this from the protocol lets a caller ask "are the inputs the
    sealed ones?" without also re-supplying the policy and consumer — so a
    drifted treatment is reported by its own named check rather than being
    swallowed as a generic fingerprint mismatch.
    """
    return {
        "records": _fingerprint([asdict(record) for record in records]),
        "questions": _fingerprint([asdict(question) for question in questions]),
        "record_embeddings": _fingerprint(_embedding_value(record_embeddings)),
        "query_embeddings": _fingerprint(_embedding_value(query_embeddings)),
    }


def policy_sha256(policy: Mapping[str, Any]) -> str:
    """Fingerprint the disposition policy by CONTENT, not by path.

    The policy is the governed arms' treatment. Revision 1 of the seal left it
    entirely outside the manifest, so an edit between seal and execution ran
    silently under a passing verify.
    """
    return _fingerprint(dict(policy))


def scoring_module_sha256() -> str:
    """Fingerprint the scorer's source. ``scoring_version`` is only a label;
    without this, a post-seal edit to score_rows ships under the sealed label.
    """
    return sha256(Path(_scoring_module.__file__).read_bytes()).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def seal_manifest(
    path: str | Path,
    records: Sequence[MemoryRecord],
    questions: Sequence[MemoryQuestion],
    record_embeddings: Mapping[str, TensorLike],
    query_embeddings: Mapping[str, TensorLike],
    retriever_settings: Mapping[str, Any],
    *,
    evidence_class: str,
    policy: Mapping[str, Any] | None = None,
    consumer_pin: str | None = None,
    registration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Seal inputs and, for a scoring run, the treatment that acts on them.

    ``evidence_class`` is mandatory and self-labelling: a ``plumbing`` seal
    refuses to carry a registration, and a ``scoring-run`` seal refuses to
    exist without a complete one. There is no default, because the dangerous
    mistake is sealing a real run as though it were a rehearsal.
    """
    if evidence_class not in EVIDENCE_CLASSES:
        raise ValueError(
            f"evidence_class must be one of {list(EVIDENCE_CLASSES)}, got {evidence_class!r}"
        )

    if evidence_class == "plumbing":
        if registration is not None:
            raise ValueError("a plumbing seal cannot carry a registration")
    else:
        _refuse_unless_registrable(retriever_settings, policy, consumer_pin, registration)

    manifest = _manifest_body(
        records,
        questions,
        record_embeddings,
        query_embeddings,
        retriever_settings,
        evidence_class=evidence_class,
        policy=policy,
        consumer_pin=consumer_pin,
        registration=registration,
    )
    manifest["manifest_sha256"] = _fingerprint(manifest)
    Path(path).write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return manifest


def _refuse_unless_registrable(
    retriever_settings: Mapping[str, Any],
    policy: Mapping[str, Any] | None,
    consumer_pin: str | None,
    registration: Mapping[str, Any] | None,
) -> None:
    """Gate G-I1 + G-I2 at seal time. Reports every reason at once."""
    problems: list[str] = []
    if policy is None:
        problems.append("scoring-run seal requires the disposition policy (G-I2)")
    if not consumer_pin:
        problems.append("scoring-run seal requires a consumer_pin (G-I2)")
    if "fam_max_entries" not in retriever_settings:
        problems.append(
            "scoring-run seal requires an explicit fam_max_entries in "
            "retriever_settings (G-I2); a data-derived default cannot be attested"
        )
    if registration is None:
        problems.append("scoring-run seal requires a registration block (G-I1)")
    else:
        problems.extend(
            f"registration: {error}" for error in validate_registration(registration)
        )
    if problems:
        raise RuntimeError(
            "refusing to seal a scoring run:\n  - " + "\n  - ".join(problems)
        )


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("sealed") is not True:
        raise RuntimeError("experiment manifest is not sealed")
    claimed = manifest.get("manifest_sha256")
    if not isinstance(claimed, str):
        raise RuntimeError("experiment manifest has no seal digest")
    body = dict(manifest)
    del body["manifest_sha256"]
    if _fingerprint(body) != claimed:
        raise RuntimeError("experiment manifest seal digest mismatch")
    return manifest


def verify_manifest(
    path: str | Path,
    records: Sequence[MemoryRecord],
    questions: Sequence[MemoryQuestion],
    record_embeddings: Mapping[str, TensorLike],
    query_embeddings: Mapping[str, TensorLike],
    retriever_settings: Mapping[str, Any],
    *,
    evidence_class: str,
    policy: Mapping[str, Any] | None = None,
    consumer_pin: str | None = None,
    registration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = load_manifest(path)
    expected = _manifest_body(
        records,
        questions,
        record_embeddings,
        query_embeddings,
        retriever_settings,
        evidence_class=evidence_class,
        policy=policy,
        consumer_pin=consumer_pin,
        registration=registration,
    )
    if manifest.get("fingerprints") != expected["fingerprints"]:
        raise RuntimeError("manifest fingerprint mismatch")
    if manifest.get("protocol") != expected["protocol"]:
        raise RuntimeError("manifest fingerprint mismatch: protocol")
    return manifest


def _manifest_body(
    records: Sequence[MemoryRecord],
    questions: Sequence[MemoryQuestion],
    record_embeddings: Mapping[str, TensorLike],
    query_embeddings: Mapping[str, TensorLike],
    retriever_settings: Mapping[str, Any],
    *,
    evidence_class: str,
    policy: Mapping[str, Any] | None,
    consumer_pin: str | None,
    registration: Mapping[str, Any] | None,
) -> dict[str, Any]:
    records_value = [asdict(record) for record in records]
    questions_value = [asdict(question) for question in questions]
    record_vectors = _embedding_value(record_embeddings)
    query_vectors = _embedding_value(query_embeddings)
    protocol = {
        "arms": list(ARM_NAMES),
        "context_budget_tokens": CONTEXT_BUDGET_TOKENS,
        "evidence_class": evidence_class,
        "scoring_version": SCORING_VERSION,
        # The scorer's source, not just its label (G-I2, scorer identity).
        "scoring_module_sha256": scoring_module_sha256(),
        "policy_version": POLICY_VERSION,
        # The governed arms' treatment (G-I2).
        "policy_sha256": None if policy is None else policy_sha256(policy),
        "consumer_pin": consumer_pin,
        "registration": None if registration is None else dict(registration),
        "retriever_settings": dict(retriever_settings),
    }
    fingerprints = {
        "records": _fingerprint(records_value),
        "questions": _fingerprint(questions_value),
        "record_embeddings": _fingerprint(record_vectors),
        "query_embeddings": _fingerprint(query_vectors),
        "protocol": _fingerprint(protocol),
    }
    return {
        "manifest_version": MANIFEST_VERSION,
        "sealed": True,
        "record_count": len(records_value),
        "question_count": len(questions_value),
        "protocol": protocol,
        "fingerprints": fingerprints,
    }


def _embedding_value(embeddings: Mapping[str, TensorLike]) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for item_id in sorted(embeddings):
        vector = torch.as_tensor(embeddings[item_id], dtype=torch.float32).detach().cpu()
        if vector.ndim != 1 or not bool(torch.isfinite(vector).all()):
            raise ValueError(f"invalid embedding for {item_id}")
        result[str(item_id)] = [float(value) for value in vector.tolist()]
    return result


def _fingerprint(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()
