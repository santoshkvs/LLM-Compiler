"""Exhaustive verification of a compiled table over its finite input space.

Because every feature is categorical and finite, the whole space can be
enumerated, which makes these checks *exact* (not sampled) for spaces up to
``max_cases``. No SMT solver is needed at this scale.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .errors import SpaceTooLargeError, VerificationError
from .schema import Invariant


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class VerificationReport:
    checks: List[Check] = field(default_factory=list)
    counterexamples: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    cases_checked: int = 0

    @property
    def ok(self) -> bool:
        return all(c.passed for c in self.checks)

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append(Check(name, passed, detail))

    def summary(self) -> str:
        lines = []
        for c in self.checks:
            lines.append(f"  [{'PASS' if c.passed else 'FAIL'}] {c.name}" + (f": {c.detail}" if c.detail else ""))
            for ex in self.counterexamples.get(c.name, []):
                lines.append(f"         counterexample: {ex}")
        for w in self.warnings:
            lines.append(f"  [WARN] {w}")
        return "\n".join(lines)

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise VerificationError(self)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "cases_checked": self.cases_checked,
            "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail} for c in self.checks],
            "warnings": list(self.warnings),
        }


def verify_table(
    table,
    *,
    approved: Optional[Mapping[str, str]] = None,
    invariants: Optional[Sequence[Invariant]] = None,
    max_cases: int = 2_000_000,
    max_counterexamples: int = 5,
) -> VerificationReport:
    """Check a table for well-formedness, completeness, consistency, invariants,
    and (optionally) equivalence with the human-approved case map."""
    invs = list(table.invariants if invariants is None else invariants)
    report = VerificationReport()
    feats = {f.name: set(f.values) for f in table.features}
    names = [f.name for f in table.features]
    outputs = set(table.outputs)

    # ---- structural well-formedness
    problems: List[str] = []
    if table.default_output not in outputs:
        problems.append(f"default output {table.default_output!r} is not a valid output")
    if table.safe_default not in outputs:
        problems.append(f"safe_default {table.safe_default!r} is not a valid output")
    ids = [r.id for r in table.rules]
    if len(set(ids)) != len(ids):
        problems.append("duplicate rule ids")
    for r in table.rules:
        if r.output not in outputs:
            problems.append(f"{r.id}: unknown output {r.output!r}")
        for f, vals in r.conditions.items():
            if f not in feats:
                problems.append(f"{r.id}: unknown feature {f!r}")
            elif not set(vals) <= feats[f]:
                problems.append(f"{r.id}: values {sorted(set(vals) - feats[f])} not in domain of {f!r}")
            elif not vals:
                problems.append(f"{r.id}: empty condition on {f!r}")
    report.add("well-formed", not problems, "; ".join(problems[:3]))
    if problems:
        return report  # enumeration below would be meaningless

    size = 1
    for f in table.features:
        size *= len(f.values)
    if size > max_cases:
        raise SpaceTooLargeError(
            f"input space has {size:,} cases (limit {max_cases:,}). Reduce/bucket features or raise max_cases."
        )

    fired = [0] * len(table.rules)
    default_fired = 0
    conflicts = 0
    conflict_examples: List[Dict[str, Any]] = []
    overlaps = 0
    inv_checked = {i.name: 0 for i in invs}
    inv_bad: Dict[str, int] = {i.name: 0 for i in invs}
    inv_examples: Dict[str, List[Dict[str, Any]]] = {i.name: [] for i in invs}
    approved_bad = 0
    approved_examples: List[Dict[str, Any]] = []
    spec_like_key = lambda c: "|".join(f"{n}={c[n]}" for n in names)  # noqa: E731

    rules = table.rules
    for combo in itertools.product(*[f.values for f in table.features]):
        case = dict(zip(names, combo))
        matching = [i for i, r in enumerate(rules) if r.matches(case)]
        if matching:
            first = matching[0]
            out = rules[first].output
            rid = rules[first].id
            fired[first] += 1
            if len(matching) > 1:
                overlaps += 1
                if len({rules[i].output for i in matching}) > 1:
                    conflicts += 1
                    if len(conflict_examples) < max_counterexamples:
                        conflict_examples.append({"case": case, "rules": [rules[i].id for i in matching]})
        else:
            out, rid = table.default_output, "default"
            default_fired += 1
        for inv in invs:
            if inv.applies(case):
                inv_checked[inv.name] += 1
                if not inv.allows(out):
                    inv_bad[inv.name] += 1
                    if len(inv_examples[inv.name]) < max_counterexamples:
                        inv_examples[inv.name].append({"case": case, "output": out, "rule": rid})
        if approved is not None:
            expected = approved.get(spec_like_key(case), table.safe_default)
            if expected != out:
                approved_bad += 1
                if len(approved_examples) < max_counterexamples:
                    approved_examples.append({"case": case, "expected": expected, "got": out})

    report.cases_checked = size
    report.add(
        "complete",
        True,
        f"all {size:,} cases produce a valid output ({default_fired:,} via default rule)",
    )

    if table.ordered:
        report.add(
            "consistent",
            True,
            f"ordered table; {overlaps:,} overlapping cases are resolved by rule order",
        )
    else:
        report.add("consistent", conflicts == 0, f"{conflicts} case(s) matched by rules with different outputs")
        if conflict_examples:
            report.counterexamples["consistent"] = conflict_examples

    dead = [rules[i].id for i, n in enumerate(fired) if n == 0]
    if dead:
        report.warnings.append(f"rules that never fire (shadowed or redundant): {dead}")

    for inv in invs:
        name = f"invariant: {inv.name}"
        detail = f"{inv_checked[inv.name]:,} case(s) in scope"
        if inv_bad[inv.name]:
            detail = f"{inv_bad[inv.name]} violation(s) in {inv_checked[inv.name]:,} case(s) in scope"
            report.counterexamples[name] = inv_examples[inv.name]
        elif inv_checked[inv.name] == 0:
            report.warnings.append(f"invariant {inv.name!r} matches no cases (vacuous)")
        report.add(name, inv_bad[inv.name] == 0, detail)

    if approved is not None:
        report.add(
            "matches approved decisions",
            approved_bad == 0,
            f"{approved_bad} disagreement(s)" if approved_bad else f"all {size:,} cases agree",
        )
        if approved_examples:
            report.counterexamples["matches approved decisions"] = approved_examples
    return report
