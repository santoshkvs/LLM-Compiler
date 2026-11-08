"""Turn raw votes into per-case verdicts, human resolutions, and an approved case map."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional

from .errors import SpecError, UnresolvedCasesError
from .sampling import VoteStore
from .schema import Spec

UNANIMOUS = "unanimous"
MAJORITY = "majority"
AMBIGUOUS = "ambiguous"
UNSAMPLED = "unsampled"


@dataclass
class CaseVerdict:
    key: str
    features: Dict[str, str]
    status: str
    proposed: Optional[str]
    agreement: float
    counts: Dict[str, int]
    invalid: int
    auto_accepted: bool


def evaluate_votes(spec: Spec, store: VoteStore, *, min_agreement: float = 1.0) -> Dict[str, CaseVerdict]:
    """Classify every case in the spec.

    ``agreement`` = (votes for the top label) / (all votes, including unparseable ones).
    A case is auto-accepted only if agreement >= ``min_agreement``. The default
    of 1.0 means only unanimous cases skip human review.
    """
    if store.spec_hash != spec.spec_hash():
        raise SpecError("votes were collected for a different spec")
    verdicts: Dict[str, CaseVerdict] = {}
    for case in spec.iter_cases():
        key = spec.case_key(case)
        rec = store.cases.get(key)
        if not rec or (sum(rec["votes"].values()) + rec["invalid"]) == 0:
            verdicts[key] = CaseVerdict(key, case, UNSAMPLED, None, 0.0, {}, 0, False)
            continue
        counts = dict(rec["votes"])
        invalid = rec["invalid"]
        total = sum(counts.values()) + invalid
        if counts:
            # deterministic tie-break: spec output order
            order = {o: i for i, o in enumerate(spec.output_names)}
            top = sorted(counts.items(), key=lambda kv: (-kv[1], order.get(kv[0], 99)))[0]
            proposed, top_n = top
            agreement = top_n / total
        else:
            proposed, agreement = None, 0.0
        if agreement == 1.0:
            status = UNANIMOUS
        elif agreement >= 0.5 and proposed is not None:
            status = MAJORITY
        else:
            status = AMBIGUOUS
        accepted = proposed is not None and agreement >= min_agreement
        verdicts[key] = CaseVerdict(key, case, status, proposed, agreement, counts, invalid, accepted)
    return verdicts


@dataclass
class Resolution:
    output: str
    reviewer: str = ""
    note: str = ""


@dataclass
class Approved:
    """The human-approved truth the compiler works from."""

    mapping: Dict[str, str] = field(default_factory=dict)
    provenance: Dict[str, str] = field(default_factory=dict)  # key -> consensus | human | safe_default
    reviewers: List[str] = field(default_factory=list)

    def provenance_counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for p in self.provenance.values():
            out[p] = out.get(p, 0) + 1
        return out


def approve(
    spec: Spec,
    verdicts: Mapping[str, CaseVerdict],
    resolutions: Optional[Mapping[str, Resolution]] = None,
    *,
    allow_unresolved: bool = False,
) -> Approved:
    """Combine auto-accepted consensus with human resolutions.

    A human resolution always wins over the model's vote (including for cases
    that were auto-accepted: that is how spot-check overrides work).
    """
    resolutions = resolutions or {}
    for k in resolutions:
        if k not in verdicts:
            raise SpecError(f"resolution for unknown case {k!r}")
    appr = Approved()
    unresolved: List[str] = []
    reviewers = set()
    for key, v in verdicts.items():
        if key in resolutions:
            r = resolutions[key]
            if r.output not in spec.outputs:
                raise SpecError(f"resolution {r.output!r} for {key!r} is not a valid output")
            appr.mapping[key] = r.output
            appr.provenance[key] = "human"
            if r.reviewer:
                reviewers.add(r.reviewer)
        elif v.auto_accepted and v.proposed is not None:
            appr.mapping[key] = v.proposed
            appr.provenance[key] = "consensus"
        elif allow_unresolved:
            appr.mapping[key] = spec.safe_default
            appr.provenance[key] = "safe_default"
        else:
            unresolved.append(key)
    if unresolved:
        raise UnresolvedCasesError(unresolved)
    appr.reviewers = sorted(reviewers)
    return appr


# ------------------------------------------------------------------ review CSV
def write_review_file(
    spec: Spec,
    verdicts: Mapping[str, CaseVerdict],
    path: str,
    *,
    include_accepted: bool = False,
) -> int:
    """Write a CSV for the policy owner. Fill the ``resolution`` column to decide a case.

    With ``include_accepted=True`` auto-accepted cases are included too, so a
    reviewer can spot-check them (and override by filling ``resolution``).
    Returns the number of rows written.
    """
    cols = ["case_key"] + spec.feature_names + [
        "status", "model_proposal", "agreement", "votes", "invalid_votes", "resolution", "reviewer", "note",
    ]
    n = 0
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for key, v in verdicts.items():
            if v.auto_accepted and not include_accepted:
                continue
            votes = "; ".join(f"{o}={c}" for o, c in sorted(v.counts.items(), key=lambda kv: -kv[1]))
            row = [key] + [v.features[f] for f in spec.feature_names] + [
                v.status, v.proposed or "", f"{v.agreement:.2f}", votes, v.invalid, "", "", "",
            ]
            w.writerow(row)
            n += 1
    return n


def read_resolutions(spec: Spec, path: str) -> Dict[str, Resolution]:
    """Read a review CSV; rows with an empty ``resolution`` are skipped."""
    out: Dict[str, Resolution] = {}
    with open(path, "r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            res = (row.get("resolution") or "").strip()
            if not res:
                continue
            if res not in spec.outputs:
                raise SpecError(f"row {row.get('case_key')!r}: resolution {res!r} is not one of {spec.output_names}")
            out[row["case_key"]] = Resolution(res, (row.get("reviewer") or "").strip(), (row.get("note") or "").strip())
    return out
