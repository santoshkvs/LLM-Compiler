# decisionforge

**Use an LLM to *write* a decision policy once, under human supervision, then run a frozen, signed, verifiable decision table with no LLM in the decision path.**

LLMs are good at judgment over fuzzy policy text and bad at being repeatable. Payments, refunds, claims, access control and moderation need the opposite at runtime: the same input must always produce the same, explainable, auditable decision.

`decisionforge` is a compiler for that workflow:

```
 BUILD TIME  (LLM + humans, run occasionally)
 Spec ──▶ Sample ──▶ Consensus ──▶ Human ──▶ Compile ──▶ Verify ──▶ Sign ──▶ table.json
 features  LLM × N    keep only     review    rules       invariants
 outputs   paraphrased unanimous    disputed  (exactly    exhaustively
 invariants prompts    answers      cases     equivalent) checked

 RUN TIME  (no LLM makes the decision)
 request ──▶ features ──▶ table.evaluate() ──▶ decision + rule id + table hash ──▶ audit log
             (known from your systems, or extracted by an LLM into a closed
              vocabulary, validated, and sent to REVIEW if anything is off)
```

> **Status: alpha (0.1.0).** The core pipeline is implemented and tested (51 tests), but it has **not** been exercised against live LLM APIs by its author, and it has not been used in production. Read [Limitations](#limitations-read-this-before-relying-on-it) first.

---

## Table of contents

1. [What problem this solves (and what it doesn't)](#what-problem-this-solves-and-what-it-doesnt)
2. [Install](#install)
3. [60-second demo (no API key)](#60-second-demo-no-api-key)
4. [How it works](#how-it-works)
5. [Step-by-step guide](#step-by-step-guide)
6. [The spec file](#the-spec-file)
7. [Runtime: engine, extraction, audit log](#runtime-engine-extraction-audit-log)
8. [What is verified, exactly](#what-is-verified-exactly)
9. [Signing and integrity](#signing-and-integrity)
10. [CLI reference](#cli-reference)
11. [Python API overview](#python-api-overview)
12. [Choosing a strategy; performance](#choosing-a-strategy-performance)
13. [Security notes](#security-notes)
14. [Limitations](#limitations-read-this-before-relying-on-it)
15. [Design decisions and prior art](#design-decisions-and-prior-art)
16. [Roadmap](#roadmap)
17. [Contributing](#contributing) · [License](#license)

---

## What problem this solves (and what it doesn't)

**Solves:** you want LLM-quality interpretation of a written policy, but the *decision* in production must be repeatable, explainable ("rule R2 of table v12"), reviewable by compliance, and unable to drift between model versions.

**The key idea:** LLM nondeterminism is not removed; it is *confined to build time*, where it can be measured (repeated, paraphrased sampling), reviewed (humans resolve disagreements), and checked (invariants) before anything is frozen.

**Does not solve:**

- **Feature extraction quality.** If the LLM (or your code) mislabels a feature, the table faithfully makes the wrong decision. The table is deterministic; it is only as correct as its inputs. See [Runtime](#runtime-engine-extraction-audit-log) for the mitigations built in.
- **Policy correctness.** The table encodes what humans approved.
- **Moving money.** This library produces a *decision*. Amount calculation, idempotency keys, ledgering and retries are your payment system's job.
- **Large or continuous input spaces.** Inputs must be bucketed into finite categorical features.

---

## Install

```bash
pip install decisionforge                 # core: no third-party dependencies
pip install "decisionforge[anthropic]"    # + Anthropic provider
pip install "decisionforge[openai]"       # + OpenAI / OpenAI-compatible provider (Ollama, vLLM, ...)
pip install "decisionforge[crypto]"       # + Ed25519 signatures
```

From source:

```bash
git clone https://github.com/YOUR_USERNAME/decisionforge && cd decisionforge
pip install -e ".[dev]" && pytest
```

Python 3.9+.

---

## 60-second demo (no API key)

```bash
decisionforge demo --out demo_out
```

This runs the whole pipeline on a refund-approval policy with a **simulated** LLM (a scripted stand-in that follows the policy but flip-flops where the policy is vague; it is not a real model). Real output:

```
[1/6] Spec: 4 features, 108 cases, outputs ['APPROVE', 'REVIEW', 'DECLINE']
[2/6] Sampled 15 votes per case (1620 simulated LLM calls)
[3/6] Consensus: 102 unanimous, 6 need human review -> demo_out/review.csv
[4/6] Approved all 108 cases; provenance {'consensus': 102, 'human': 6}
[5/6] Compiled 108 cases into 6 rules + default
      verification: well-formed=PASS, complete=PASS, consistent=PASS, invariant: ...=PASS (x3),
      matches approved decisions=PASS
[6/6] Signed and saved -> demo_out/refund_policy_v1.json

refund_approval v1  (first match wins)
  R1  days_since_purchase=OVER90 AND customer_history in {GOOD, NEW}  ->  DECLINE
  R2  amount_band in {LOW, MID} AND reason in {DEFECTIVE, NOT_RECEIVED} AND customer_history=GOOD  ->  APPROVE
  R3  amount_band in {LOW, MID} AND days_since_purchase=D90 AND reason=CHANGED_MIND
        AND customer_history in {GOOD, NEW}  ->  DECLINE
  R4  amount_band=LOW AND days_since_purchase=D30 AND reason in {DEFECTIVE, CHANGED_MIND}
        AND customer_history in {GOOD, NEW}  ->  APPROVE
  R5  amount_band in {LOW, MID} AND days_since_purchase=D30 AND reason=NOT_AS_DESCRIBED
        AND customer_history=GOOD  ->  APPROVE
  R6  amount_band=MID AND days_since_purchase=D30 AND reason=DEFECTIVE AND customer_history=NEW  ->  APPROVE
  else  REVIEW
```

108 cases became 6 rules plus a default. Six disputed cases went to a human; 102 unanimous ones did not need one.

### The safety net: a confident, unanimous mistake

Unanimous agreement between samples does not mean the model is *right*. Run:

```bash
decisionforge demo --out demo_bad --inject-violation
```

Here the simulated LLM consistently approves a **HIGH-value** refund, so consensus filtering lets it through. The invariant check stops the build:

```
[5/6] Compile + verify: FAILED (this is the safety net working)
  [FAIL] invariant: high-value refunds are never auto-approved: 1 violation(s) in 36 case(s) in scope
         counterexample: {'case': {'amount_band': 'HIGH', 'days_since_purchase': 'D30',
                                   'reason': 'DEFECTIVE', 'customer_history': 'GOOD'}, 'output': 'APPROVE', 'rule': 'R7'}
```

No table is produced. That is the point of declaring invariants.

---

## How it works

| Stage | Module | What happens |
|---|---|---|
| **Spec** | `schema` | You declare features (finite, categorical), outputs, a `safe_default`, the policy text, and **invariants** (safety properties that must always hold). |
| **Sample** | `sampling` | For every case in the input space, the LLM is asked `templates × samples_per_prompt` times (default 3 × 5 = 15), using *paraphrased* prompts. Votes are stored and resumable. |
| **Consensus** | `consensus` | Cases where every vote agrees are auto-accepted. Everything else (disagreement, unparseable answers) goes to a human. `min_agreement` lets you loosen this. |
| **Review** | `consensus` | A CSV is written for the policy owner, who fills a `resolution` column. Human resolutions always override model votes, including for auto-accepted cases (spot-check with `--include-accepted`). Unresolved cases block compilation unless you explicitly map them to `safe_default`. |
| **Compile** | `compiler` | The approved case map is compressed into rules, either as an ordered *decision list* (compact, readable) or as *disjoint rules* from a decision tree (fast, order-independent). The result is **checked to be exactly equivalent** to the approved map. |
| **Verify** | `verify` | Exhaustive checks over the whole input space: well-formedness, completeness, consistency, your invariants, and equivalence to the approved decisions. |
| **Freeze** | `table`, `signing` | The table is saved as JSON with a content hash (tamper detection) and an optional HMAC or Ed25519 signature. |
| **Run** | `engine`, `extract` | `table.evaluate(features)` is a pure function. Optionally an LLM maps text to features (never to decisions), with validation and fail-safe fallback. Every decision can go to a JSONL audit log. |

---

## Step-by-step guide

### 1. Write a spec

`spec.json` (full example in [`examples/refund/spec.json`](examples/refund/spec.json)):

```json
{
  "name": "refund_approval",
  "policy_file": "policy.txt",
  "features": [
    {"name": "amount_band", "values": ["LOW", "MID", "HIGH"],
     "value_descriptions": {"LOW": "under $50", "MID": "$50 to $500", "HIGH": "over $500"}},
    {"name": "customer_history", "values": ["GOOD", "NEW", "FLAGGED"]}
  ],
  "outputs": {"APPROVE": "refund automatically", "REVIEW": "send to a human", "DECLINE": "reject"},
  "safe_default": "REVIEW",
  "invariants": [
    {"name": "high-value never auto-approved", "when": {"amount_band": ["HIGH"]}, "never": ["APPROVE"]}
  ]
}
```

### 2. Sample the LLM

```bash
export ANTHROPIC_API_KEY=...
decisionforge sample --spec spec.json --provider anthropic --model <model-name> --out votes.json
```

Cost is `cases × templates × samples_per_prompt` calls (printed before it starts). It checkpoints to `votes.json` and resumes if interrupted. Use `--provider openai --base-url http://localhost:11434/v1` for local OpenAI-compatible servers. A small, cheap model is usually enough, since answers are single labels.

### 3. Review disagreements

```bash
decisionforge review --spec spec.json --votes votes.json --out review.csv
```

Open `review.csv`. Each row shows the case, the vote breakdown (e.g. `APPROVE=10; REVIEW=5`) and blank `resolution`, `reviewer`, `note` columns. Fill them in. Add `--include-accepted` to also see the unanimous cases and override any of them.

### 4. Compile and verify

```bash
decisionforge compile --spec spec.json --votes votes.json --resolutions review.csv \
                      --table-version 1 --out refund_policy_v1.json
```

Compilation fails if any case is unresolved or any invariant is violated, and prints counterexamples.

### 5. Sign

```bash
export DF_SIGNING_KEY=$(decisionforge keygen --alg hmac-sha256)   # store in a secret manager
decisionforge sign --table refund_policy_v1.json --key-env DF_SIGNING_KEY
```

For Ed25519 (sign with a private key in CI, verify with only the public key in production):

```bash
decisionforge keygen --alg ed25519      # prints a private and a public key (hex)
decisionforge sign --table refund_policy_v1.json --alg ed25519 --key-env DF_PRIVATE_KEY
```

### 6. Use it

```python
from decisionforge import DecisionTable

table = DecisionTable.load("refund_policy_v1.json", verify_key=KEY, require_signature=True)
d = table.evaluate({"amount_band": "LOW", "days_since_purchase": "D30",
                    "reason": "DEFECTIVE", "customer_history": "GOOD"})
d.output      # 'APPROVE'
d.rule_id     # 'R2'
d.table_hash  # sha256 of the table content, for the audit trail
```

`KEY` is the HMAC secret or the Ed25519 **public** key. The whole pipeline in Python is in [`examples/refund/run_pipeline.py`](examples/refund/run_pipeline.py).

---

## The spec file

| Field | Required | Meaning |
|---|---|---|
| `name` | yes | Table name (appears in audit records). |
| `features` | yes | List of `{name, values, description?, value_descriptions?}`. Values are non-empty strings; names cannot contain `\|` or `=`. Descriptions are shown to the LLM, so make bucket boundaries explicit. |
| `outputs` | yes | List of labels, or `{label: description}`. At least two. |
| `safe_default` | yes | The fail-safe output (usually a human-review label). Used for unresolved cases (if allowed), unmapped cases, and failed extraction. |
| `policy` / `policy_file` | recommended | The policy text shown to the LLM. `policy_file` is resolved relative to the spec file. |
| `invariants` | recommended | `{name, when: {feature: [values]}, never: [outputs] and/or only: [outputs]}`. Within the `when` region the output must not be in `never` and, if `only` is given, must be in `only`. An empty `when` means "everywhere". |

The number of cases is the product of the feature cardinalities. Keep it to what you can review: tens to low thousands is comfortable.

---

## Runtime: engine, extraction, audit log

```python
from decisionforge import (DecisionEngine, FeatureExtractor, AuditLogger,
                           AnthropicProvider, DecisionTable, Spec)

spec = Spec.from_json("spec.json")
table = DecisionTable.load("refund_policy_v1.json", verify_key=KEY, require_signature=True)
extractor = FeatureExtractor(spec, AnthropicProvider("<model>", max_tokens=300), min_confidence=0.85)
engine = DecisionEngine(table, extractor, AuditLogger("decisions.jsonl"))

d = engine.decide_text(
    "My headphones arrived broken, I want my money back",
    known_features={"amount_band": "LOW", "customer_history": "GOOD"},   # from YOUR systems, not the LLM
    request_id="req-123",
)
```

Built-in protections around the one place an LLM still touches a live request:

- **Prefer deterministic features.** Anything in `known_features` (amounts, account age, flags) is used as given, is *never* asked of the LLM, and cannot be overridden by it.
- **Closed vocabulary.** The LLM must return a value from each feature's allowed set; anything else fails.
- **Fail closed.** Invalid JSON, a missing feature, an out-of-domain value, a missing or low confidence score, or a provider exception all return `safe_default` with `fallback=True` and a `reason`. The engine never guesses.
- **The LLM never sees or produces the decision.** At worst, adversarial text changes *which allowed feature values* are extracted, which is why high-stakes features should come from your systems.
- **Audit log.** One JSON line per decision: table name, version and hash, features, matched rule, fallback flag and reason, confidence. Input text is stored only as a SHA-256 hash by default; pass `log_input=True` to store it (mind personal data).

`engine.decide(features)` skips extraction and is a pure function of its inputs.

---

## What is verified, exactly

`verify_table` enumerates **every** case in the input space, so results are exact, not sampled. (This is feasible because features are finite and categorical; no SMT solver is needed at this scale.)

| Check | Meaning |
|---|---|
| `well-formed` | Rules reference real features, values and outputs; no duplicate ids; default is valid. |
| `complete` | Every possible input yields a valid output (guaranteed by the default rule; also reports how many cases fall to it). |
| `consistent` | For *disjoint* tables: no case is matched by rules with different outputs. For *ordered* tables: overlaps are reported and resolved by rule order. |
| `invariant: …` | For every case in the invariant's region, the output is allowed. Reports counterexamples. Warns if an invariant matches no cases (vacuous). |
| `matches approved decisions` | The compiled table agrees with the human-approved map on all cases. |
| warnings | Rules that never fire (redundant or shadowed). |

Loading a table also checks its **content hash**, so any edit to the file is detected (`IntegrityError`).

What verification does *not* tell you: that the approved decisions are right. Only that the table faithfully implements them and satisfies the invariants you wrote.

---

## Signing and integrity

- Every saved table has a `content_hash`: SHA-256 over canonical JSON of everything that affects behavior. Timestamps and metadata are excluded, so recompiling identical input gives an identical hash.
- `sign_table` signs that hash with **HMAC-SHA256** (shared secret, stdlib only) or **Ed25519** (`pip install decisionforge[crypto]`).
- `DecisionTable.load(path, verify_key=..., require_signature=True)` rejects unsigned, tampered or wrongly-signed tables.
- Key management is yours: keep keys in a secret manager, not in the repo. The `demo` command uses a hard-coded secret that must never be used for real.

---

## CLI reference

| Command | Purpose |
|---|---|
| `decisionforge demo [--out DIR] [--inject-violation]` | Offline end-to-end demo with a simulated LLM. |
| `decisionforge sample --spec S --provider {anthropic,openai} --model M [--base-url U] [--samples-per-prompt 5] [--temperature 0.7] [--workers 4] [--out votes.json]` | Collect LLM votes (resumable). |
| `decisionforge review --spec S --votes V [--min-agreement 1.0] [--include-accepted] [--out review.csv]` | Write the human-review CSV. |
| `decisionforge compile --spec S --votes V [--resolutions R] [--allow-unresolved] [--strategy {auto,decision_list,tree}] [--default-strategy {safe,majority,best}] [--table-version N] [--out table.json]` | Compile + verify. |
| `decisionforge verify --table T [--key-env VAR] [--require-signature]` | Re-verify a table file. Exit code 1 on failure. |
| `decisionforge show --table T` | Print rules. |
| `decisionforge decide --table T (--features JSON \| --features-file F) [--key-env VAR] [--require-signature]` | Evaluate one case. |
| `decisionforge keygen [--alg {ed25519,hmac-sha256}]` | Generate a key. |
| `decisionforge sign --table T --key-env VAR [--alg ...] [--key-id ID] [--out F]` | Sign a table. |

Exit codes: `0` success, `1` verification/demo failure, `2` other errors.

---

## Python API overview

```python
from decisionforge import (
    Spec, Feature, Invariant,                         # define the problem
    Sampler, VoteStore, CallableProvider,             # build-time sampling (+ AnthropicProvider, OpenAIProvider)
    evaluate_votes, write_review_file, read_resolutions, approve, Resolution,   # consensus + review
    compile_table, verify_table,                      # compile and verify
    DecisionTable, sign_table, verify_table_signature, generate_ed25519_keypair,
    FeatureExtractor, DecisionEngine, AuditLogger,    # runtime
)
```

Bring your own LLM backend by wrapping any function:

```python
provider = CallableProvider(lambda prompt, temperature, seed: my_llm(prompt))
```

Custom prompt paraphrases: `Sampler(spec, provider, templates=[...])`. Templates use the literal placeholders `{policy}`, `{features}`, `{case}`, `{outputs}`.

All errors derive from `DecisionForgeError` (`SpecError`, `UnresolvedCasesError`, `VerificationError`, `IntegrityError`, `SignatureError`, `InvalidInputError`, ...).

---

## Choosing a strategy; performance

| `--strategy` | Rules | Table type | Speed (measured, single core) |
|---|---|---|---|
| `decision_list` | ordered, first match wins; usually fewest and most readable | `ordered=True` | roughly quadratic: ~0.4 s at 625 cases, ~5 s at 3,100, ~12 s at 4,100 |
| `tree` | disjoint (order does not matter) | `ordered=False`, verified conflict-free | ~0.8 s at 100,000 cases, ~2.4 s at 262,000 |
| `auto` (default) | `decision_list` up to 2,500 cases, otherwise `tree` | | |

Both are greedy: neither guarantees the *minimum* number of rules, but both are verified to be *exactly* equivalent to the approved decisions. Timings are from a synthetic policy in a small container; yours will vary.

`--default-strategy`: `safe` (default: the final `else` is your `safe_default`, so anything not explicitly approved falls to human review), `majority`, or `best` (tries every output as the default and keeps the fewest rules).

Enumeration limits: `compile_table(max_cases=500_000)` and `verify_table(max_cases=2_000_000)`. Larger spaces raise `SpaceTooLargeError`.

---

## Security notes

- **Untrusted text only ever feeds feature extraction**, and its output is validated against a closed vocabulary. Still, treat extraction as attackable: a customer can try to talk the LLM into a favorable feature value. Source high-stakes features from your own systems via `known_features`, and use `invariants` as guard rails.
- **Sampling prompts contain only your policy and the case**, never user text.
- **Table files are data, not code.** Loading one never executes anything. Verify hash and signature before trusting it.
- **Don't put secrets or personal data in the spec or policy**; they are sent to your LLM provider during sampling.
- Before publishing, add a `SECURITY.md` with a private contact for vulnerability reports.

---

## Limitations (read this before relying on it)

1. **Finite, categorical inputs only.** Numbers, free text and dates must be bucketed into a closed set *before* the table. Boundary choices (is $500 MID or HIGH?) are your responsibility and a common source of real-world bugs.
2. **Combinatorial growth.** Five features with ten values each is 100,000 cases: too many to review by hand and expensive to sample. Use few, coarse features, or split the policy into several tables.
3. **Consensus is not correctness.** An LLM can be consistently wrong. Mitigations: invariants, human spot-checks (`--include-accepted`), and a clear policy. The library cannot detect a confident error that doesn't violate an invariant.
4. **Extraction is the weak link at runtime.** Test your extractor on labeled examples; this version ships no evaluation harness for it.
5. **Not independently audited, and not exercised against live LLM APIs by the author.** The Anthropic and OpenAI adapters are thin wrappers that were not run against real endpoints during development. Test them with your own keys; SDKs and model parameter rules change over time.
6. **Greedy minimization:** rule sets are correct but not provably minimal.
7. **No migration tooling yet.** A policy change means recompiling with a new version; compare tables with `show` or by diffing the JSON.
8. **A decision tool, not a payment system.** Idempotency, authorization, ledgering and regulatory compliance are out of scope, and using this library does not make you compliant with anything by itself.

---

## Design decisions and prior art

The individual ideas here are not new, and this project doesn't claim they are: decision tables and DMN, rule induction and distilling models into decision trees, and policy-as-code engines all exist. What this library packages is the *workflow* that connects them for LLM-authored policy: paraphrase-consistency sampling, human resolution of disagreements, exact compilation, exhaustive invariant checking, a signed artifact, and a fail-safe runtime. If you need a general rules engine or policy language, use one of those and consider generating its input from the verified table.

Notable choices:

- **Exhaustive enumeration instead of an SMT solver.** With finite domains it is exact, simple and dependency-free. It stops scaling beyond a few million cases.
- **Human resolutions beat the model**, always, including over unanimous votes.
- **Fail toward `safe_default`** everywhere (unmapped cases, unresolved cases if allowed, failed extraction).
- **Zero required dependencies**; SDKs are optional extras.
- **Tables are JSON**, so they can be diffed, reviewed in pull requests, and loaded from other languages.

---

## Roadmap

Roughly in priority order (contributions welcome):

- Extraction evaluation harness (labeled messages → accuracy, confusion matrix, paraphrase robustness)
- Table diff/migration report between versions ("these 14 cases changed from APPROVE to REVIEW")
- Optional SMT backend (z3) and pairwise/sampled coverage for spaces too large to enumerate
- Better rule minimization
- Export to other formats (DMN, OPA/Rego, Cedar, plain Python)
- Hierarchical/composed tables
- Async providers and more provider adapters
- Pluggable key management for signing

---

## Contributing

```bash
pip install -e ".[dev]"
pytest
```

Please add tests with changes. Good first issues: new provider adapters, the extraction evaluation harness, and rule-minimization improvements. This project deliberately avoids becoming a general rules engine; please keep issues within that scope.

## License

MIT. See [LICENSE](LICENSE).
