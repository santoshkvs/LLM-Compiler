"""decisionforge: compile approved LLM judgment into a frozen, verifiable decision table.

Build time (LLM + humans)            Run time (no LLM in the decision)
  Spec -> Sampler -> consensus  ->  review  ->  compile  ->  verify  ->  sign  ->  DecisionTable.evaluate()
"""

__version__ = "0.1.0"

from .compiler import compile_table
from .consensus import (
    Approved,
    CaseVerdict,
    Resolution,
    approve,
    evaluate_votes,
    read_resolutions,
    write_review_file,
)
from .engine import AuditLogger, DecisionEngine
from .errors import (
    CompilationError,
    DecisionForgeError,
    IntegrityError,
    InvalidInputError,
    ProviderError,
    SignatureError,
    SpaceTooLargeError,
    SpecError,
    UnresolvedCasesError,
    VerificationError,
)
from .extract import Extraction, FeatureExtractor
from .providers import AnthropicProvider, CallableProvider, OpenAIProvider, Provider, RetryingProvider
from .sampling import DEFAULT_TEMPLATES, Sampler, VoteStore, parse_decision, render_prompt
from .schema import Feature, Invariant, Spec
from .signing import generate_ed25519_keypair, sign_table, verify_table_signature
from .table import Decision, DecisionTable, Rule
from .verify import VerificationReport, verify_table

__all__ = [
    "__version__",
    "Spec", "Feature", "Invariant",
    "Provider", "CallableProvider", "AnthropicProvider", "OpenAIProvider", "RetryingProvider",
    "Sampler", "VoteStore", "DEFAULT_TEMPLATES", "render_prompt", "parse_decision",
    "evaluate_votes", "CaseVerdict", "Resolution", "Approved", "approve",
    "write_review_file", "read_resolutions",
    "compile_table", "DecisionTable", "Rule", "Decision",
    "verify_table", "VerificationReport",
    "sign_table", "verify_table_signature", "generate_ed25519_keypair",
    "FeatureExtractor", "Extraction", "DecisionEngine", "AuditLogger",
    "DecisionForgeError", "SpecError", "SpaceTooLargeError", "ProviderError", "UnresolvedCasesError",
    "CompilationError", "VerificationError", "IntegrityError", "SignatureError", "InvalidInputError",
]
