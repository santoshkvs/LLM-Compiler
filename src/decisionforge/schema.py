"""Spec: the closed decision space (features, outputs, invariants)."""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Mapping, Optional

from .errors import SpecError

_FORBIDDEN_CHARS = ("|", "\n", "\r")


def canonical_json(obj: Any) -> str:
    """Deterministic JSON used for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class Feature:
    """A categorical input feature with a finite, closed set of values."""

    name: str
    values: List[str]
    description: str = ""
    value_descriptions: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"name": self.name, "values": list(self.values)}
        if self.description:
            d["description"] = self.description
        if self.value_descriptions:
            d["value_descriptions"] = dict(self.value_descriptions)
        return d


@dataclass
class Invariant:
    """A safety property that must hold for every case in a region.

    ``when`` selects a region (feature -> allowed values; omitted features are
    unconstrained). Within that region the table's output must not be in
    ``never`` and, if ``only`` is non-empty, must be in ``only``.
    """

    name: str
    when: Dict[str, List[str]] = field(default_factory=dict)
    never: List[str] = field(default_factory=list)
    only: List[str] = field(default_factory=list)
    description: str = ""

    def __post_init__(self) -> None:
        self._when_sets = {f: frozenset(v) for f, v in self.when.items()}

    def applies(self, case: Mapping[str, str]) -> bool:
        for f, vals in self._when_sets.items():
            if case[f] not in vals:
                return False
        return True

    def allows(self, output: str) -> bool:
        if output in self.never:
            return False
        if self.only and output not in self.only:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"name": self.name, "when": {k: list(v) for k, v in self.when.items()}}
        if self.never:
            d["never"] = list(self.never)
        if self.only:
            d["only"] = list(self.only)
        if self.description:
            d["description"] = self.description
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Invariant":
        when = {k: ([v] if isinstance(v, str) else list(v)) for k, v in (d.get("when") or {}).items()}
        return cls(
            name=d["name"],
            when=when,
            never=list(d.get("never") or []),
            only=list(d.get("only") or []),
            description=d.get("description", ""),
        )


@dataclass
class Spec:
    """Everything the compiler needs to know about a decision problem."""

    name: str
    features: List[Feature]
    outputs: Dict[str, str]
    safe_default: str
    policy: str = ""
    description: str = ""
    invariants: List[Invariant] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.validate()

    # ------------------------------------------------------------------ basics
    @property
    def feature_names(self) -> List[str]:
        return [f.name for f in self.features]

    @property
    def output_names(self) -> List[str]:
        return list(self.outputs)

    def feature(self, name: str) -> Feature:
        for f in self.features:
            if f.name == name:
                return f
        raise KeyError(name)

    @property
    def size(self) -> int:
        n = 1
        for f in self.features:
            n *= len(f.values)
        return n

    def iter_cases(self) -> Iterator[Dict[str, str]]:
        names = self.feature_names
        for combo in itertools.product(*[f.values for f in self.features]):
            yield dict(zip(names, combo))

    def case_key(self, case: Mapping[str, str]) -> str:
        return "|".join(f"{f.name}={case[f.name]}" for f in self.features)

    def parse_case_key(self, key: str) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for part in key.split("|"):
            k, _, v = part.partition("=")
            out[k] = v
        if list(out) != self.feature_names:
            raise SpecError(f"case key {key!r} does not match spec features")
        return out

    # -------------------------------------------------------------- validation
    def validate(self) -> None:
        if not self.name:
            raise SpecError("spec needs a name")
        if not self.features:
            raise SpecError("spec needs at least one feature")
        seen = set()
        for f in self.features:
            if not f.name or any(c in f.name for c in _FORBIDDEN_CHARS + ("=",)):
                raise SpecError(f"invalid feature name {f.name!r}")
            if f.name in seen:
                raise SpecError(f"duplicate feature {f.name!r}")
            seen.add(f.name)
            if not f.values:
                raise SpecError(f"feature {f.name!r} has no values")
            if len(set(f.values)) != len(f.values):
                raise SpecError(f"feature {f.name!r} has duplicate values")
            for v in f.values:
                if not isinstance(v, str) or not v or any(c in v for c in _FORBIDDEN_CHARS):
                    raise SpecError(f"feature {f.name!r} has invalid value {v!r} (must be non-empty strings)")
        if len(self.outputs) < 2:
            raise SpecError("spec needs at least two outputs")
        if self.safe_default not in self.outputs:
            raise SpecError(f"safe_default {self.safe_default!r} is not one of the outputs")
        for inv in self.invariants:
            for fname, vals in inv.when.items():
                if fname not in seen:
                    raise SpecError(f"invariant {inv.name!r} references unknown feature {fname!r}")
                bad = [v for v in vals if v not in self.feature(fname).values]
                if bad:
                    raise SpecError(f"invariant {inv.name!r}: unknown values {bad} for {fname!r}")
            for o in list(inv.never) + list(inv.only):
                if o not in self.outputs:
                    raise SpecError(f"invariant {inv.name!r} references unknown output {o!r}")
            if not inv.never and not inv.only:
                raise SpecError(f"invariant {inv.name!r} needs 'never' and/or 'only'")

    # ------------------------------------------------------------ serialization
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "policy": self.policy,
            "features": [f.to_dict() for f in self.features],
            "outputs": dict(self.outputs),
            "safe_default": self.safe_default,
            "invariants": [i.to_dict() for i in self.invariants],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Spec":
        raw_features = d.get("features")
        if not raw_features:
            raise SpecError("spec needs 'features'")
        features: List[Feature] = []
        if isinstance(raw_features, Mapping):
            items = [dict(name=k, **(v if isinstance(v, Mapping) else {"values": v})) for k, v in raw_features.items()]
        else:
            items = list(raw_features)
        for it in items:
            features.append(
                Feature(
                    name=it["name"],
                    values=[str(v) for v in it["values"]],
                    description=it.get("description", ""),
                    value_descriptions=dict(it.get("value_descriptions") or {}),
                )
            )
        raw_outputs = d.get("outputs")
        if not raw_outputs:
            raise SpecError("spec needs 'outputs'")
        if isinstance(raw_outputs, Mapping):
            outputs = {str(k): str(v) for k, v in raw_outputs.items()}
        else:
            outputs = {str(k): "" for k in raw_outputs}
        if "safe_default" not in d:
            raise SpecError("spec needs 'safe_default' (the output used when unsure, e.g. REVIEW)")
        return cls(
            name=d.get("name", "unnamed"),
            features=features,
            outputs=outputs,
            safe_default=d["safe_default"],
            policy=d.get("policy", ""),
            description=d.get("description", ""),
            invariants=[Invariant.from_dict(i) for i in d.get("invariants", [])],
        )

    @classmethod
    def from_json(cls, path: str) -> "Spec":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        # allow "policy_file" relative to the spec file
        if "policy_file" in data and not data.get("policy"):
            import os

            p = os.path.join(os.path.dirname(os.path.abspath(path)), data["policy_file"])
            with open(p, "r", encoding="utf-8") as pf:
                data["policy"] = pf.read()
        return cls.from_dict(data)

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)
            fh.write("\n")

    def spec_hash(self) -> str:
        return sha256_hex(canonical_json(self.to_dict()))
