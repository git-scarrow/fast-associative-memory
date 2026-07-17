"""Fixed-budget evaluation of governed and ungoverned agent memory."""

ARM_NAMES = (
    "no_memory",
    "exemplar_raw",
    "exemplar_governed",
    "fam_raw",
    "fam_governed",
)
CONTEXT_BUDGET_TOKENS = 1500

__all__ = ["ARM_NAMES", "CONTEXT_BUDGET_TOKENS"]
