"""Self-contained refund-approval demo. Runs offline with a *simulated* LLM.

The simulated provider is a stand-in so you can see the whole pipeline (and its
failure modes) without an API key. It is NOT a real model.
"""

from __future__ import annotations

import csv
import hashlib
import os
import random
import re
from typing import Dict, Optional

from .compiler import compile_table
from .consensus import approve, evaluate_votes, read_resolutions, write_review_file
from .errors import VerificationError
from .providers import Provider
from .sampling import Sampler
from .schema import Spec
from .signing import sign_table
from .table import DecisionTable

REFUND_POLICY = """\
REFUND POLICY (excerpt)
1. Customers flagged for abuse are never auto-refunded; a human must look at the request.
2. Requests made more than 90 days after purchase are declined.
3. Refunds above $500 always need human review.
4. Defective items returned within 30 days are refunded. Between 31 and 90 days, refund only
   low- or mid-value orders from customers in good standing; otherwise a human decides.
5. Items not received are refunded for customers in good standing; new customers go to review.
6. Items not as described within 30 days are refunded for customers in good standing;
   for new customers the case is a judgment call and should go to review.
7. Change-of-mind refunds are allowed only within 30 days and only for low-value orders.
   Mid-value change-of-mind requests are a judgment call. After 30 days they are declined.
"""

REFUND_SPEC = {
    "name": "refund_approval",
    "description": "Decide whether a refund request is approved automatically, reviewed by a human, or declined.",
    "policy": REFUND_POLICY,
    "features": [
        {"name": "amount_band", "values": ["LOW", "MID", "HIGH"], "description": "order value",
         "value_descriptions": {"LOW": "under $50", "MID": "$50 to $500", "HIGH": "over $500"}},
        {"name": "days_since_purchase", "values": ["D30", "D90", "OVER90"], "description": "time since purchase",
         "value_descriptions": {"D30": "0-30 days", "D90": "31-90 days", "OVER90": "more than 90 days"}},
        {"name": "reason", "values": ["DEFECTIVE", "NOT_AS_DESCRIBED", "NOT_RECEIVED", "CHANGED_MIND"],
         "description": "customer's stated reason"},
        {"name": "customer_history", "values": ["GOOD", "NEW", "FLAGGED"], "description": "standing of the customer",
         "value_descriptions": {"GOOD": "established, no abuse", "NEW": "first orders", "FLAGGED": "flagged for abuse"}},
    ],
    "outputs": {
        "APPROVE": "refund automatically",
        "REVIEW": "send to a human agent",
        "DECLINE": "reject the request",
    },
    "safe_default": "REVIEW",
    "invariants": [
        {"name": "high-value refunds are never auto-approved", "when": {"amount_band": ["HIGH"]}, "never": ["APPROVE"]},
        {"name": "flagged customers are never auto-approved", "when": {"customer_history": ["FLAGGED"]}, "never": ["APPROVE"]},
        {"name": "nothing after 90 days is approved", "when": {"days_since_purchase": ["OVER90"]}, "never": ["APPROVE"]},
    ],
}


def refund_spec() -> Spec:
    return Spec.from_dict(REFUND_SPEC)


def _ideal(c: Dict[str, str]) -> str:
    if c["customer_history"] == "FLAGGED":
        return "REVIEW"
    if c["days_since_purchase"] == "OVER90":
        return "DECLINE"
    if c["amount_band"] == "HIGH":
        return "REVIEW"
    r, d, a, h = c["reason"], c["days_since_purchase"], c["amount_band"], c["customer_history"]
    if r == "DEFECTIVE":
        if d == "D30":
            return "APPROVE"
        return "APPROVE" if h == "GOOD" else "REVIEW"
    if r == "NOT_RECEIVED":
        return "APPROVE" if h == "GOOD" else "REVIEW"
    if r == "NOT_AS_DESCRIBED":
        return "APPROVE" if (d == "D30" and h == "GOOD") else "REVIEW"
    if r == "CHANGED_MIND":
        if d == "D30" and a == "LOW":
            return "APPROVE"
        return "DECLINE" if d == "D90" else "REVIEW"
    return "REVIEW"


def _wobbly(c: Dict[str, str]) -> bool:
    """Cases the policy leaves genuinely open: the simulated LLM flip-flops on these."""
    return (
        (c["reason"] == "NOT_AS_DESCRIBED" and c["days_since_purchase"] == "D30" and c["customer_history"] == "NEW")
        or (c["reason"] == "CHANGED_MIND" and c["days_since_purchase"] == "D30" and c["amount_band"] == "MID")
    )


class SimulatedRefundProvider(Provider):
    """A fake 'LLM' for demos: mostly follows the policy, flip-flops where the policy is vague.

    With ``inject_violation=True`` it *consistently* approves one high-value
    case: the kind of confident mistake that unanimity alone cannot catch, but
    the invariant check does.
    """

    def __init__(self, inject_violation: bool = False):
        self.inject_violation = inject_violation

    def complete(self, prompt: str, *, temperature: float = 0.0, seed: Optional[int] = None) -> str:
        names = [f["name"] for f in REFUND_SPEC["features"]]
        case = {k: v for k, v in re.findall(r"^- (\w+): (\S+)", prompt, flags=re.M) if k in names}
        ideal = _ideal(case)
        if self.inject_violation and case == {
            "amount_band": "HIGH", "days_since_purchase": "D30", "reason": "DEFECTIVE", "customer_history": "GOOD",
        }:
            return "APPROVE"
        if _wobbly(case):
            rng = random.Random(int(hashlib.sha256(f"{prompt}|{seed}".encode()).hexdigest(), 16))
            return rng.choice(["APPROVE", "REVIEW"]) if ideal != "DECLINE" else rng.choice(["DECLINE", "REVIEW"])
        return ideal


def run_demo(outdir: str, *, inject_violation: bool = False, say=print) -> Optional[DecisionTable]:
    os.makedirs(outdir, exist_ok=True)
    spec = refund_spec()
    spec.to_json(os.path.join(outdir, "spec.json"))
    say(f"[1/6] Spec: {len(spec.features)} features, {spec.size} cases, outputs {spec.output_names}")

    sampler = Sampler(spec, SimulatedRefundProvider(inject_violation), samples_per_prompt=5)
    votes = sampler.run()
    votes.save(os.path.join(outdir, "votes.json"))
    say(f"[2/6] Sampled {sampler.votes_per_case} votes per case ({sampler.votes_per_case * spec.size} simulated LLM calls)")

    verdicts = evaluate_votes(spec, votes, min_agreement=1.0)
    accepted = sum(v.auto_accepted for v in verdicts.values())
    review_path = os.path.join(outdir, "review.csv")
    n = write_review_file(spec, verdicts, review_path)
    say(f"[3/6] Consensus: {accepted} unanimous, {n} need human review -> {review_path}")

    # Simulate the policy owner: fill the 'resolution' column conservatively (REVIEW).
    with open(review_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        row["resolution"] = "REVIEW"
        row["reviewer"] = "demo-policy-owner"
        row["note"] = "policy leaves this open; route to a human"
    with open(review_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())) if rows else None
        if w:
            w.writeheader()
            w.writerows(rows)
    resolutions = read_resolutions(spec, review_path)
    approved = approve(spec, verdicts, resolutions)
    say(f"[4/6] Approved all {len(approved.mapping)} cases; provenance {approved.provenance_counts()}")

    try:
        table = compile_table(spec, approved, version="1")
    except VerificationError as exc:
        say("[5/6] Compile + verify: FAILED (this is the safety net working)")
        say(exc.report.summary())
        return None
    say(f"[5/6] Compiled {spec.size} cases into {len(table.rules)} rules + default")
    checks = table.metadata["verification"]["checks"]
    say("      verification: " + ", ".join(f"{c['name']}={'PASS' if c['passed'] else 'FAIL'}" for c in checks))

    sign_table(table, "demo-secret-do-not-use-in-production", key_id="demo")
    path = os.path.join(outdir, "refund_policy_v1.json")
    table.save(path)
    say(f"[6/6] Signed and saved -> {path}")
    say("")
    say(table.to_text())
    return table
