"""Command-line interface: ``decisionforge <command>``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from . import __version__
from .compiler import DEFAULT_STRATEGIES, STRATEGIES, compile_table
from .consensus import approve, evaluate_votes, read_resolutions, write_review_file
from .errors import DecisionForgeError
from .providers import AnthropicProvider, OpenAIProvider, RetryingProvider
from .sampling import Sampler, VoteStore
from .schema import Spec
from .signing import ED25519, HMAC, generate_ed25519_keypair, sign_table, verify_table_signature
from .table import DecisionTable
from .verify import verify_table


def _key_from_env(var: str) -> str:
    val = os.environ.get(var)
    if not val:
        raise DecisionForgeError(f"environment variable {var} is not set")
    return val


def _cmd_demo(a) -> int:
    from .demo import run_demo

    t = run_demo(a.out, inject_violation=a.inject_violation)
    return 0 if t is not None else 1


def _cmd_sample(a) -> int:
    spec = Spec.from_json(a.spec)
    if a.provider == "anthropic":
        provider = AnthropicProvider(a.model)
    else:
        provider = OpenAIProvider(a.model, base_url=a.base_url)
    provider = RetryingProvider(provider)
    store = VoteStore.load(a.out) if os.path.exists(a.out) else None
    sampler = Sampler(
        spec, provider, samples_per_prompt=a.samples_per_prompt, temperature=a.temperature, workers=a.workers,
        progress=lambda d, t: print(f"\r  sampled {d}/{t} cases", end="", file=sys.stderr),
    )
    n = sampler.votes_per_case
    print(f"{spec.size} cases x {n} votes = {spec.size * n} LLM calls (resumable; checkpointing to {a.out})", file=sys.stderr)
    store = sampler.run(store, checkpoint=a.out)
    print("", file=sys.stderr)
    store.save(a.out)
    print(f"wrote {a.out}")
    return 0


def _cmd_review(a) -> int:
    spec = Spec.from_json(a.spec)
    verdicts = evaluate_votes(spec, VoteStore.load(a.votes), min_agreement=a.min_agreement)
    n = write_review_file(spec, verdicts, a.out, include_accepted=a.include_accepted)
    accepted = sum(v.auto_accepted for v in verdicts.values())
    print(f"{accepted} case(s) auto-accepted by consensus; wrote {n} row(s) to {a.out}")
    print("Fill the 'resolution' column (and 'reviewer') for each case that needs a decision.")
    return 0


def _cmd_compile(a) -> int:
    spec = Spec.from_json(a.spec)
    verdicts = evaluate_votes(spec, VoteStore.load(a.votes), min_agreement=a.min_agreement)
    resolutions = read_resolutions(spec, a.resolutions) if a.resolutions else {}
    approved = approve(spec, verdicts, resolutions, allow_unresolved=a.allow_unresolved)
    table = compile_table(spec, approved, version=a.table_version, strategy=a.strategy, default_strategy=a.default_strategy)
    table.save(a.out)
    print(f"compiled {spec.size} cases into {len(table.rules)} rules -> {a.out}")
    print(f"content hash: {table.content_hash()}")
    print(f"provenance: {approved.provenance_counts()}")
    return 0


def _cmd_verify(a) -> int:
    key = _key_from_env(a.key_env) if a.key_env else None
    table = DecisionTable.load(a.table, verify_key=key, require_signature=a.require_signature)
    report = verify_table(table)
    print(f"{table.name} v{table.version}  hash={table.content_hash()[:16]}...")
    if key is not None:
        print("  [PASS] signature valid")
    print(report.summary())
    return 0 if report.ok else 1


def _cmd_show(a) -> int:
    print(DecisionTable.load(a.table).to_text())
    return 0


def _cmd_decide(a) -> int:
    key = _key_from_env(a.key_env) if a.key_env else None
    table = DecisionTable.load(a.table, verify_key=key, require_signature=a.require_signature)
    if a.features_file:
        with open(a.features_file, "r", encoding="utf-8") as fh:
            feats = json.load(fh)
    else:
        feats = json.loads(a.features)
    print(json.dumps(table.evaluate(feats).to_record(), indent=2))
    return 0


def _cmd_keygen(a) -> int:
    if a.alg == ED25519:
        priv, pub = generate_ed25519_keypair()
        print(f"private key (keep secret): {priv}")
        print(f"public key  (distribute) : {pub}")
    else:
        print(os.urandom(32).hex())
    return 0


def _cmd_sign(a) -> int:
    table = DecisionTable.load(a.table)
    sign_table(table, _key_from_env(a.key_env), alg=a.alg, key_id=a.key_id)
    out = a.out or a.table
    table.save(out)
    print(f"signed ({a.alg}) -> {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="decisionforge", description=__doc__)
    p.add_argument("--version", action="version", version=f"decisionforge {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="run the offline refund demo with a simulated LLM")
    d.add_argument("--out", default="decisionforge_demo")
    d.add_argument("--inject-violation", action="store_true", help="make the simulated LLM confidently approve a HIGH-value case")
    d.set_defaults(fn=_cmd_demo)

    s = sub.add_parser("sample", help="query an LLM about every case (resumable)")
    s.add_argument("--spec", required=True)
    s.add_argument("--provider", choices=["anthropic", "openai"], required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--base-url", help="OpenAI-compatible endpoint (Ollama, vLLM, ...)")
    s.add_argument("--samples-per-prompt", type=int, default=5)
    s.add_argument("--temperature", type=float, default=0.7)
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--out", default="votes.json")
    s.set_defaults(fn=_cmd_sample)

    r = sub.add_parser("review", help="write a CSV of cases that need human review")
    r.add_argument("--spec", required=True)
    r.add_argument("--votes", required=True)
    r.add_argument("--min-agreement", type=float, default=1.0)
    r.add_argument("--include-accepted", action="store_true", help="also list auto-accepted cases for spot-checking")
    r.add_argument("--out", default="review.csv")
    r.set_defaults(fn=_cmd_review)

    c = sub.add_parser("compile", help="compile approved decisions into a verified table")
    c.add_argument("--spec", required=True)
    c.add_argument("--votes", required=True)
    c.add_argument("--resolutions", help="the filled-in review CSV")
    c.add_argument("--min-agreement", type=float, default=1.0)
    c.add_argument("--allow-unresolved", action="store_true", help="map unresolved cases to safe_default instead of failing")
    c.add_argument("--strategy", choices=STRATEGIES, default="auto")
    c.add_argument("--default-strategy", choices=DEFAULT_STRATEGIES, default="safe")
    c.add_argument("--table-version", default="1")
    c.add_argument("--out", default="table.json")
    c.set_defaults(fn=_cmd_compile)

    v = sub.add_parser("verify", help="re-verify a table file (hash, optional signature, all checks)")
    v.add_argument("--table", required=True)
    v.add_argument("--key-env", help="env var holding the HMAC secret or Ed25519 public key (hex)")
    v.add_argument("--require-signature", action="store_true")
    v.set_defaults(fn=_cmd_verify)

    sh = sub.add_parser("show", help="print a table's rules")
    sh.add_argument("--table", required=True)
    sh.set_defaults(fn=_cmd_show)

    de = sub.add_parser("decide", help="evaluate a table on explicit features")
    de.add_argument("--table", required=True)
    g = de.add_mutually_exclusive_group(required=True)
    g.add_argument("--features", help='JSON object, e.g. \'{"amount_band":"LOW",...}\'')
    g.add_argument("--features-file")
    de.add_argument("--key-env")
    de.add_argument("--require-signature", action="store_true")
    de.set_defaults(fn=_cmd_decide)

    k = sub.add_parser("keygen", help="generate a signing key")
    k.add_argument("--alg", choices=[HMAC, ED25519], default=ED25519)
    k.set_defaults(fn=_cmd_keygen)

    sg = sub.add_parser("sign", help="sign a table file")
    sg.add_argument("--table", required=True)
    sg.add_argument("--key-env", required=True, help="env var holding the HMAC secret or Ed25519 private key (hex)")
    sg.add_argument("--alg", choices=[HMAC, ED25519], default=HMAC)
    sg.add_argument("--key-id")
    sg.add_argument("--out", help="default: overwrite --table")
    sg.set_defaults(fn=_cmd_sign)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except DecisionForgeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
