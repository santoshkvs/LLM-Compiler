"""The full decisionforge pipeline, step by step, using the Python API.

Runs offline with a simulated LLM. To use a real model, replace the provider
with ``AnthropicProvider("<model>")`` or ``OpenAIProvider("<model>")`` (see run_with_llm.py).

    python examples/refund/run_pipeline.py
"""
import csv
import os
import tempfile

from decisionforge import (
    Sampler, Spec, approve, compile_table, evaluate_votes,
    read_resolutions, sign_table, write_review_file,
)
from decisionforge.demo import SimulatedRefundProvider

here = os.path.dirname(os.path.abspath(__file__))
spec = Spec.from_json(os.path.join(here, "spec.json"))          # 1. the closed decision space
print(f"{spec.size} cases to decide")

votes = Sampler(spec, SimulatedRefundProvider(), samples_per_prompt=5).run()   # 2. ask the LLM (15 votes/case)

verdicts = evaluate_votes(spec, votes, min_agreement=1.0)       # 3. keep only unanimous answers automatically
needs_review = [v for v in verdicts.values() if not v.auto_accepted]
print(f"{len(needs_review)} case(s) need a human:")
for v in needs_review:
    print("  ", v.features, "-> votes", v.counts)

workdir = tempfile.mkdtemp()
review_csv = os.path.join(workdir, "review.csv")
write_review_file(spec, verdicts, review_csv)                   # 4. hand the disputed cases to the policy owner
# ... in real life a person edits review.csv here. We simulate them choosing REVIEW:
rows = list(csv.DictReader(open(review_csv)))
for r in rows:
    r.update(resolution="REVIEW", reviewer="alice@example.com", note="policy is silent; escalate")
with open(review_csv, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

approved = approve(spec, verdicts, read_resolutions(spec, review_csv))
table = compile_table(spec, approved, version="1")              # 5. compile + verify invariants (raises on failure)
print(table.to_text())

sign_table(table, os.environ.get("DF_SIGNING_KEY", "dev-only-secret"), key_id="example")   # 6. sign and ship
table.save(os.path.join(workdir, "refund_policy_v1.json"))

decision = table.evaluate({                                      # 7. runtime: no LLM involved
    "amount_band": "LOW", "days_since_purchase": "D30",
    "reason": "DEFECTIVE", "customer_history": "GOOD",
})
print(decision.output, "via", decision.rule_id, "table", decision.table_hash[:12])
