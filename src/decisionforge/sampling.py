"""Build-time LLM sampling: ask the model about every case, several ways, several times."""

from __future__ import annotations

import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence

from .errors import SpecError
from .providers import Provider
from .schema import Spec

# Paraphrased prompt templates. Consistency across paraphrases is the point:
# a decision that flips when the wording changes is not a decision to freeze.
# Placeholders are replaced literally (no str.format) so policy text may contain braces.
DEFAULT_TEMPLATES: List[str] = [
    (
        "You are applying a written policy to one case.\n\n"
        "POLICY:\n{policy}\n\n"
        "FEATURE DEFINITIONS:\n{features}\n\n"
        "CASE:\n{case}\n\n"
        "ALLOWED DECISIONS:\n{outputs}\n\n"
        "Reply with exactly one decision label and nothing else."
    ),
    (
        "Decide what the policy below requires for the case described.\n\n"
        "CASE:\n{case}\n\n"
        "FEATURE DEFINITIONS:\n{features}\n\n"
        "POLICY:\n{policy}\n\n"
        "Choose one of:\n{outputs}\n\n"
        "Answer with the label only."
    ),
    (
        "As a careful policy analyst, determine the correct outcome.\n"
        "Policy:\n{policy}\n\n"
        "Case details:\n{case}\n\n"
        "Definitions:\n{features}\n\n"
        "Possible outcomes:\n{outputs}\n\n"
        "If the policy does not clearly settle the case, still pick the single most defensible label. "
        "Output just that label."
    ),
]


def render_prompt(template: str, spec: Spec, case: Dict[str, str]) -> str:
    case_lines = []
    for f in spec.features:
        v = case[f.name]
        extra = f.value_descriptions.get(v, "")
        case_lines.append(f"- {f.name}: {v}" + (f" ({extra})" if extra else ""))
    feat_lines = []
    for f in spec.features:
        d = f" - {f.description}" if f.description else ""
        feat_lines.append(f"- {f.name}{d}")
        for v in f.values:
            vd = f.value_descriptions.get(v)
            if vd:
                feat_lines.append(f"    {v}: {vd}")
    out_lines = [f"- {o}" + (f": {d}" if d else "") for o, d in spec.outputs.items()]
    return (
        template.replace("{policy}", spec.policy or "(no policy text provided)")
        .replace("{features}", "\n".join(feat_lines))
        .replace("{case}", "\n".join(case_lines))
        .replace("{outputs}", "\n".join(out_lines))
    )


def parse_decision(text: str, outputs: Sequence[str]) -> Optional[str]:
    """Return the single output label found in ``text``, or None if zero or several.

    Accepts a bare label, a JSON object like {"decision": "APPROVE"}, or a short sentence.
    """
    text = (text or "").strip()
    if not text:
        return None
    if text.startswith("{"):
        try:
            obj = json.loads(text)
            for key in ("decision", "label", "output", "answer"):
                if isinstance(obj, dict) and obj.get(key) in outputs:
                    return obj[key]
        except ValueError:
            pass
    head = text[:200]
    found = []
    for o in outputs:
        pat = r"(?<![A-Za-z0-9_])" + re.escape(o) + r"(?![A-Za-z0-9_])"
        if re.search(pat, head, flags=re.IGNORECASE):
            found.append(o)
    return found[0] if len(found) == 1 else None


class VoteStore:
    """Raw votes per case, persisted as JSON so long runs can resume."""

    def __init__(self, spec_hash: str, config: Optional[Dict[str, Any]] = None):
        self.spec_hash = spec_hash
        self.config = config or {}
        self.cases: Dict[str, Dict[str, Any]] = {}

    def total(self, key: str) -> int:
        c = self.cases.get(key)
        return 0 if not c else sum(c["votes"].values()) + c["invalid"]

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"spec_hash": self.spec_hash, "config": self.config, "cases": self.cases}, fh, indent=1)

    @classmethod
    def load(cls, path: str) -> "VoteStore":
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        s = cls(d["spec_hash"], d.get("config"))
        s.cases = d["cases"]
        return s


class Sampler:
    """Query a provider ``len(templates) * samples_per_prompt`` times per case."""

    def __init__(
        self,
        spec: Spec,
        provider: Provider,
        *,
        templates: Optional[Sequence[str]] = None,
        samples_per_prompt: int = 5,
        temperature: float = 0.7,
        workers: int = 1,
        progress: Optional[Callable[[int, int], None]] = None,
    ):
        self.spec = spec
        self.provider = provider
        self.templates = list(templates or DEFAULT_TEMPLATES)
        self.samples_per_prompt = samples_per_prompt
        self.temperature = temperature
        self.workers = max(1, workers)
        self.progress = progress

    @property
    def votes_per_case(self) -> int:
        return len(self.templates) * self.samples_per_prompt

    def _sample_case(self, case: Dict[str, str]) -> Dict[str, Any]:
        votes: Counter = Counter()
        invalid = 0
        examples: List[str] = []
        outputs = self.spec.output_names
        for ti, tpl in enumerate(self.templates):
            prompt = render_prompt(tpl, self.spec, case)
            for s in range(self.samples_per_prompt):
                raw = self.provider.complete(prompt, temperature=self.temperature, seed=ti * 1000 + s)
                label = parse_decision(raw, outputs)
                if label is None:
                    invalid += 1
                    if len(examples) < 3:
                        examples.append((raw or "")[:120])
                else:
                    votes[label] += 1
        return {"votes": dict(votes), "invalid": invalid, "invalid_examples": examples}

    def run(
        self,
        store: Optional[VoteStore] = None,
        *,
        checkpoint: Optional[str] = None,
        checkpoint_every: int = 25,
    ) -> VoteStore:
        spec = self.spec
        if store is None:
            store = VoteStore(
                spec.spec_hash(),
                {"templates": len(self.templates), "samples_per_prompt": self.samples_per_prompt,
                 "temperature": self.temperature},
            )
        elif store.spec_hash != spec.spec_hash():
            raise SpecError("these votes were collected for a different spec; start a new vote file")

        todo = [c for c in spec.iter_cases() if store.total(spec.case_key(c)) < self.votes_per_case]
        done = 0
        total = len(todo)

        def work(case: Dict[str, str]):
            return spec.case_key(case), self._sample_case(case)

        try:
            if self.workers == 1:
                results = map(work, todo)
                for key, res in results:
                    store.cases[key] = res
                    done += 1
                    if self.progress:
                        self.progress(done, total)
                    if checkpoint and done % checkpoint_every == 0:
                        store.save(checkpoint)
            else:
                with ThreadPoolExecutor(max_workers=self.workers) as pool:
                    for key, res in pool.map(work, todo):
                        store.cases[key] = res
                        done += 1
                        if self.progress:
                            self.progress(done, total)
                        if checkpoint and done % checkpoint_every == 0:
                            store.save(checkpoint)
        finally:
            if checkpoint:
                store.save(checkpoint)
        return store
