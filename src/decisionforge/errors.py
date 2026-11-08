"""Exception hierarchy for decisionforge."""

from __future__ import annotations


class DecisionForgeError(Exception):
    """Base class for all decisionforge errors."""


class SpecError(DecisionForgeError, ValueError):
    """The spec (features, outputs, invariants) is malformed."""


class SpaceTooLargeError(DecisionForgeError):
    """The input space is larger than the configured enumeration limit."""


class ProviderError(DecisionForgeError):
    """An LLM provider is missing, misconfigured, or failed."""


class UnresolvedCasesError(DecisionForgeError):
    """Cases that need human review have not been resolved."""

    def __init__(self, keys):
        self.keys = list(keys)
        super().__init__(
            f"{len(self.keys)} case(s) need human review and have no resolution. "
            f"Resolve them in the review file, or pass allow_unresolved=True to map "
            f"them to the spec's safe_default. First few: {self.keys[:3]}"
        )


class CompilationError(DecisionForgeError):
    """The compiler produced a table that does not match the approved cases."""


class VerificationError(DecisionForgeError):
    """A compiled table failed verification (invariant, consistency, ...)."""

    def __init__(self, report):
        self.report = report
        super().__init__("Table verification failed:\n" + report.summary())


class IntegrityError(DecisionForgeError):
    """A table file's content does not match its recorded hash."""


class SignatureError(DecisionForgeError):
    """A table signature is missing, malformed, or does not verify."""


class InvalidInputError(DecisionForgeError, ValueError):
    """Features passed to a table are missing, extra, or out of domain."""
