"""Build-time sampling and run-time extraction with a real LLM.

Requires:  pip install "decisionforge[anthropic]"  and  ANTHROPIC_API_KEY.
NOTE: this makes (cases x 15) API calls. For the refund spec that is 1,620 short calls.
Pick a small, cheap model for sampling and check current model names in your provider's docs.
"""
import os
import sys

from decisionforge import (
    AnthropicProvider, DecisionEngine, FeatureExtractor, RetryingProvider,
    Sampler, Spec, DecisionTable,
)

MODEL = os.environ.get("DF_MODEL") or sys.exit("set DF_MODEL to a model name your account can use")
here = os.path.dirname(os.path.abspath(__file__))
spec = Spec.from_json(os.path.join(here, "spec.json"))

# --- build time -----------------------------------------------------------
sampler = Sampler(spec, RetryingProvider(AnthropicProvider(MODEL)), samples_per_prompt=5, workers=4)
votes = sampler.run(checkpoint="votes.json")      # resumable: rerun to continue after an interruption
print("votes saved to votes.json; now run:  decisionforge review --spec examples/refund/spec.json --votes votes.json")

# --- run time (after you have compiled and signed refund_policy_v1.json) ---
if os.path.exists("refund_policy_v1.json"):
    table = DecisionTable.load("refund_policy_v1.json", verify_key=os.environ["DF_SIGNING_KEY"], require_signature=True)
    extractor = FeatureExtractor(spec, AnthropicProvider(MODEL, max_tokens=300), min_confidence=0.85)
    engine = DecisionEngine(table, extractor)
    # amount_band comes from the order system, not from the LLM:
    d = engine.decide_text("My headphones arrived broken, I want my money back. Bought them 2 weeks ago.",
                           known_features={"amount_band": "LOW", "customer_history": "GOOD"})
    print(d.output, d.rule_id, "fallback" if d.fallback else "")
