"""The only admissible way to execute a sealed scoring run.

Gate G-I2. Revision 1 of the harness left the seal *advisory*: `verify_manifest`
checked the caller-supplied dict against the file, not the configuration that
actually ran, and its sole non-test caller was the dry run. A typo'd
`candidate_k`, an edited disposition policy, a re-quantized consumer, or a
post-seal change to `score_rows` all executed under a passing verify.

Here the manifest is the source of truth: retrievers and runner are constructed
*from* the sealed protocol rather than checked against it, and every binding is
asserted before the consumer is ever called. `preflight` reports every failure
at once; `build_sealed_run` refuses to hand back a runner unless all pass.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .ledger import MemoryLedger
from .manifest import (
    input_fingerprints,
    load_manifest,
    policy_sha256,
    scoring_module_sha256,
)
from .models import MemoryQuestion, MemoryRecord
from .preregistration import validate_registration
from .retrievers import ExactVectorRetriever, FAMRetriever, TensorLike
from .runner import FiveArmRunner
from .scoring import SCORING_VERSION


@dataclass(frozen=True, slots=True)
class Check:
    gate: str
    name: str
    passed: bool
    detail: str


class PreflightFailed(RuntimeError):
    """Raised instead of returning a runner. Carries every failed check."""

    def __init__(self, checks: Sequence[Check]) -> None:
        self.checks = tuple(checks)
        self.failures = tuple(c for c in self.checks if not c.passed)
        lines = "\n  - ".join(f"[{c.gate}] {c.name}: {c.detail}" for c in self.failures)
        super().__init__(f"pre-execution verification BLOCKED:\n  - {lines}")


def preflight(
    manifest_path: str | Path,
    *,
    records: Sequence[MemoryRecord],
    questions: Sequence[MemoryQuestion],
    record_embeddings: Mapping[str, TensorLike],
    query_embeddings: Mapping[str, TensorLike],
    retriever_settings: Mapping[str, Any],
    policy: Mapping[str, Any],
    consumer: Any,
) -> tuple[Check, ...]:
    """Run every pre-execution check and report all outcomes.

    Never raises on a failed gate — it reports. Reporting all failures at once
    matters here: a run that is blocked for four reasons should disclose four,
    not the first one the code happened to reach.
    """
    checks: list[Check] = []

    # The seal digest is the one check that must pass before anything else can
    # be read: a tampered file makes every later comparison meaningless.
    try:
        manifest = load_manifest(manifest_path)
    except Exception as exc:  # noqa: BLE001 — the reason is the report
        return (
            Check("G-I2", "manifest integrity", False, f"{type(exc).__name__}: {exc}"),
        )

    protocol = manifest.get("protocol", {})

    # Inputs are checked WITHOUT re-supplying the treatment, so a drifted
    # policy or consumer is reported by its own named check below rather than
    # collapsing into a generic fingerprint mismatch here.
    sealed_inputs = {
        key: value
        for key, value in manifest.get("fingerprints", {}).items()
        if key != "protocol"
    }
    try:
        live_inputs = input_fingerprints(
            records, questions, record_embeddings, query_embeddings
        )
        drifted = sorted(
            key for key, value in live_inputs.items() if sealed_inputs.get(key) != value
        )
        checks.append(
            Check(
                "G-I2",
                "input integrity",
                not drifted,
                "records, questions and embeddings match the seal"
                if not drifted
                else f"drifted: {', '.join(drifted)}",
            )
        )
    except Exception as exc:  # noqa: BLE001
        checks.append(
            Check("G-I2", "input integrity", False, f"{type(exc).__name__}: {exc}")
        )

    sealed_settings = protocol.get("retriever_settings", {})
    checks.append(
        Check(
            "G-I2",
            "settings match the seal",
            dict(sealed_settings) == dict(retriever_settings),
            f"sealed {dict(sealed_settings)!r} vs live {dict(retriever_settings)!r}",
        )
    )

    evidence_class = protocol.get("evidence_class")
    checks.append(
        Check(
            "G-I2",
            "evidence class",
            evidence_class == "scoring-run",
            f"sealed as {evidence_class!r}; only 'scoring-run' may execute for evidence",
        )
    )

    registration = protocol.get("registration")
    if registration is None:
        checks.append(
            Check("G-I1", "registration complete", False, "no registration block sealed")
        )
    else:
        errors = validate_registration(registration)
        checks.append(
            Check(
                "G-I1",
                "registration complete",
                not errors,
                "clean" if not errors else "; ".join(errors),
            )
        )

    sealed_policy = protocol.get("policy_sha256")
    live_policy = policy_sha256(policy)
    checks.append(
        Check(
            "G-I2",
            "policy binding",
            sealed_policy == live_policy,
            f"sealed {_short(sealed_policy)} vs live {_short(live_policy)}",
        )
    )

    sealed_pin = protocol.get("consumer_pin")
    live_pin = getattr(consumer, "pin_id", None)
    checks.append(
        Check(
            "G-I2",
            "consumer pin",
            bool(sealed_pin) and sealed_pin == live_pin,
            f"sealed {sealed_pin!r} vs live {live_pin!r}",
        )
    )

    sealed_scorer = protocol.get("scoring_module_sha256")
    live_scorer = scoring_module_sha256()
    checks.append(
        Check(
            "G-I2",
            "scorer identity",
            sealed_scorer == live_scorer,
            f"sealed {_short(sealed_scorer)} vs live {_short(live_scorer)}"
            f" (version label {protocol.get('scoring_version')!r} / {SCORING_VERSION!r})",
        )
    )

    settings = protocol.get("retriever_settings", {})
    checks.append(
        Check(
            "G-I2",
            "fam capacity attested",
            "fam_max_entries" in settings,
            f"fam_max_entries={settings.get('fam_max_entries')!r}",
        )
    )

    return tuple(checks)


def build_sealed_run(
    manifest_path: str | Path,
    *,
    records: Sequence[MemoryRecord],
    questions: Sequence[MemoryQuestion],
    record_embeddings: Mapping[str, TensorLike],
    query_embeddings: Mapping[str, TensorLike],
    retriever_settings: Mapping[str, Any],
    policy: Mapping[str, Any],
    consumer: Any,
) -> FiveArmRunner:
    """Return a runner built FROM the sealed protocol, or refuse.

    Raises before the consumer is touched, so a blocked run costs no
    generation and produces no rows that could later be mistaken for evidence.
    """
    checks = preflight(
        manifest_path,
        records=records,
        questions=questions,
        record_embeddings=record_embeddings,
        query_embeddings=query_embeddings,
        retriever_settings=retriever_settings,
        policy=policy,
        consumer=consumer,
    )
    if any(not check.passed for check in checks):
        raise PreflightFailed(checks)

    sealed = load_manifest(manifest_path)["protocol"]["retriever_settings"]
    return FiveArmRunner(
        ledger=MemoryLedger(records),
        vector_retriever=ExactVectorRetriever(records, record_embeddings),
        fam_retriever=FAMRetriever(
            records,
            record_embeddings,
            prototype_k=sealed["fam_prototype_k"],
            max_entries=sealed["fam_max_entries"],
        ),
        consumer=consumer,
        candidate_k=sealed["candidate_k"],
        policy=dict(policy),
    )


def _short(digest: Any) -> str:
    return "<none>" if not isinstance(digest, str) else digest[:12]
