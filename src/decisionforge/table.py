"""The frozen artifact: a decision table with rules, hashing, and (de)serialization."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from .errors import IntegrityError, InvalidInputError, SignatureError
from .schema import Feature, Invariant, canonical_json, sha256_hex

FORMAT = "decisionforge-table/1"


@dataclass
class Rule:
    """``IF every condition holds THEN output``. Missing features are unconstrained."""

    id: str
    conditions: Dict[str, List[str]]
    output: str

    def __post_init__(self) -> None:
        self._sets = {f: frozenset(v) for f, v in self.conditions.items()}

    def matches(self, case: Mapping[str, str]) -> bool:
        for f, vals in self._sets.items():
            if case[f] not in vals:
                return False
        return True

    def describe(self) -> str:
        if not self.conditions:
            parts = "ALWAYS"
        else:
            bits = []
            for f, vals in self.conditions.items():
                bits.append(f"{f}={vals[0]}" if len(vals) == 1 else f"{f} in {{{', '.join(vals)}}}")
            parts = " AND ".join(bits)
        return f"{parts}  ->  {self.output}"

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "if": {k: list(v) for k, v in self.conditions.items()}, "then": self.output}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Rule":
        return cls(id=d["id"], conditions={k: list(v) for k, v in d["if"].items()}, output=d["then"])


@dataclass
class Decision:
    """The result of evaluating a table (plus provenance for the audit log)."""

    output: str
    rule_id: str
    rule_index: Optional[int]
    table_name: str
    table_version: str
    table_hash: str
    features: Dict[str, str]
    fallback: bool = False
    reason: Optional[str] = None
    confidence: Optional[float] = None

    def to_record(self) -> Dict[str, Any]:
        return {
            "output": self.output,
            "rule_id": self.rule_id,
            "table": {"name": self.table_name, "version": self.table_version, "hash": self.table_hash},
            "features": dict(self.features),
            "fallback": self.fallback,
            "reason": self.reason,
            "confidence": self.confidence,
        }


@dataclass
class DecisionTable:
    """A compiled, immutable-by-convention decision table.

    ``ordered=True``  -> first matching rule wins (a decision list).
    ``ordered=False`` -> rules are intended to be disjoint (verified by ``verify_table``).
    """

    name: str
    version: str
    features: List[Feature]
    outputs: List[str]
    safe_default: str
    default_output: str
    rules: List[Rule]
    ordered: bool = True
    invariants: List[Invariant] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    signature: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        self._domains = {f.name: frozenset(f.values) for f in self.features}
        self._hash: Optional[str] = None

    # --------------------------------------------------------------- evaluation
    @property
    def feature_names(self) -> List[str]:
        return [f.name for f in self.features]

    def _validate_case(self, case: Mapping[str, str]) -> None:
        missing = [n for n in self._domains if n not in case]
        extra = [n for n in case if n not in self._domains]
        if missing or extra:
            raise InvalidInputError(f"features must match the table exactly (missing={missing}, extra={extra})")
        for n, dom in self._domains.items():
            if case[n] not in dom:
                raise InvalidInputError(f"value {case[n]!r} is not allowed for feature {n!r}")

    def evaluate(self, case: Mapping[str, str], *, validate: bool = True) -> Decision:
        if validate:
            self._validate_case(case)
        for i, rule in enumerate(self.rules):
            if rule.matches(case):
                return self._decision(case, rule.output, rule.id, i)
        return self._decision(case, self.default_output, "default", None)

    def _decision(self, case, output, rule_id, idx) -> Decision:
        return Decision(
            output=output,
            rule_id=rule_id,
            rule_index=idx,
            table_name=self.name,
            table_version=self.version,
            table_hash=self.content_hash(),
            features={n: case[n] for n in self.feature_names},
        )

    # ------------------------------------------------------------------ hashing
    def content_dict(self) -> Dict[str, Any]:
        """Everything that affects behavior. Excludes timestamps and signatures."""
        return {
            "format": FORMAT,
            "name": self.name,
            "version": self.version,
            "ordered": self.ordered,
            "features": [f.to_dict() for f in self.features],
            "outputs": list(self.outputs),
            "safe_default": self.safe_default,
            "default_output": self.default_output,
            "rules": [r.to_dict() for r in self.rules],
            "invariants": [i.to_dict() for i in self.invariants],
        }

    def content_hash(self) -> str:
        if self._hash is None:
            self._hash = sha256_hex(canonical_json(self.content_dict()))
        return self._hash

    # ------------------------------------------------------------- serialization
    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": FORMAT,
            "content": self.content_dict(),
            "content_hash": self.content_hash(),
            "metadata": self.metadata,
            "signature": self.signature,
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)
            fh.write("\n")

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "DecisionTable":
        if d.get("format") != FORMAT:
            raise IntegrityError(f"unsupported table format {d.get('format')!r}")
        c = d["content"]
        feats = [
            Feature(
                name=f["name"],
                values=list(f["values"]),
                description=f.get("description", ""),
                value_descriptions=dict(f.get("value_descriptions") or {}),
            )
            for f in c["features"]
        ]
        table = cls(
            name=c["name"],
            version=c["version"],
            features=feats,
            outputs=list(c["outputs"]),
            safe_default=c["safe_default"],
            default_output=c["default_output"],
            rules=[Rule.from_dict(r) for r in c["rules"]],
            ordered=bool(c.get("ordered", True)),
            invariants=[Invariant.from_dict(i) for i in c.get("invariants", [])],
            metadata=dict(d.get("metadata") or {}),
            signature=d.get("signature"),
        )
        if table.content_hash() != d.get("content_hash"):
            raise IntegrityError("table content does not match its recorded hash (file was modified or corrupted)")
        return table

    @classmethod
    def load(
        cls,
        path: str,
        *,
        verify_key: Any = None,
        require_signature: bool = False,
    ) -> "DecisionTable":
        """Load a table, checking its content hash and (optionally) its signature.

        ``verify_key`` is the HMAC secret or the Ed25519 public key (hex).
        """
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        table = cls.from_dict(data)
        if verify_key is not None or require_signature:
            from .signing import verify_table_signature

            if table.signature is None:
                raise SignatureError("table is not signed")
            if verify_key is None:
                raise SignatureError("a verify_key is required to check the signature")
            if not verify_table_signature(table, verify_key):
                raise SignatureError("signature verification failed")
        return table

    # ------------------------------------------------------------------ display
    def to_text(self) -> str:
        mode = "first match wins" if self.ordered else "rules are disjoint"
        lines = [
            f"{self.name} v{self.version}  (hash {self.content_hash()[:12]}..., {mode})",
            f"features: " + ", ".join(f"{f.name}[{len(f.values)}]" for f in self.features),
            "",
        ]
        for r in self.rules:
            lines.append(f"  {r.id:<5} {r.describe()}")
        lines.append(f"  {'else':<5} {self.default_output}")
        return "\n".join(lines)
