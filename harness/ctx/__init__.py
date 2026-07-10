"""PR-13 governed context compiler — shared library (build checkpoint).

Registered by docs/PR13_GOVERNED_CONTEXT_COMPILER.md (r3). Analysis
/harness code only: no FAM-core import, no engine change, no serving.
"""

from harness.ctx.compile import compile, load_policy, render_raw_matched, summarize
from harness.ctx.output_contract import parse_consumer_output

__all__ = [
    "compile",
    "load_policy",
    "render_raw_matched",
    "summarize",
    "parse_consumer_output",
]
