"""Compile an approved case map into a minimal, verified decision table."""

from __future__ import annotations

import datetime as _dt
import itertools
import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

from .consensus import Approved
from .errors import CompilationError, SpaceTooLargeError, SpecError
from .schema import Spec
from .table import DecisionTable, Rule
from .verify import verify_table

Case = Tuple[str, ...]
Cube = Tuple[frozenset, ...]  # one set of allowed values per feature (full domain = unconstrained)

STRATEGIES = ("auto", "decision_list", "tree")
AUTO_LIST_LIMIT = 2500  # decision_list search is roughly quadratic; above this "auto" uses the tree
DEFAULT_STRATEGIES = ("safe", "majority", "best")


# ------------------------------------------------------------------ tree strategy
def _entropy_score(groups: Sequence[Sequence[Tuple[Case, str]]]) -> float:
    score = 0.0
    for g in groups:
        n = len(g)
        counts = Counter(o for _, o in g)
        h = -sum((c / n) * math.log2(c / n) for c in counts.values())
        score += n * h
    return score


def _build_tree(
    cases: List[Tuple[Case, str]],
    domains: List[List[str]],
    path: Dict[int, str],
    out: List[Tuple[Dict[int, str], str]],
) -> None:
    outs = {o for _, o in cases}
    if len(outs) == 1:
        out.append((dict(path), next(iter(outs))))
        return
    best: Optional[Tuple[float, int, Dict[str, List[Tuple[Case, str]]]]] = None
    for fi in range(len(domains)):
        if fi in path or len(domains[fi]) < 2:
            continue
        groups: Dict[str, List[Tuple[Case, str]]] = defaultdict(list)
        for c, o in cases:
            groups[c[fi]].append((c, o))
        score = _entropy_score(list(groups.values()))
        if best is None or score < best[0]:
            best = (score, fi, groups)
    assert best is not None, "impure leaf with no features left (duplicate cases?)"
    _, fi, groups = best
    for v in domains[fi]:
        if v in groups:
            path[fi] = v
            _build_tree(groups[v], domains, path, out)
            del path[fi]


def _merge_cubes(cubes: List[Tuple[Cube, str]]) -> List[Tuple[Cube, str]]:
    """Repeatedly union cubes with the same output that differ in exactly one feature."""
    if not cubes:
        return cubes
    nf = len(cubes[0][0])
    while True:
        merged_any = False
        for j in range(nf):
            groups: Dict[Tuple[str, Tuple[frozenset, ...]], List[frozenset]] = {}
            for cube, out in cubes:
                groups.setdefault((out, cube[:j] + cube[j + 1 :]), []).append(cube[j])
            new: List[Tuple[Cube, str]] = []
            for (out, rest), vals in groups.items():
                if len(vals) > 1:
                    merged_any = True
                new.append((rest[:j] + (frozenset().union(*vals),) + rest[j:], out))
            cubes = new
        if not merged_any:
            return cubes


def _tree_cubes(full: Dict[Case, str], domains: List[List[str]], default: str) -> List[Tuple[Cube, str]]:
    leaves: List[Tuple[Dict[int, str], str]] = []
    _build_tree(list(full.items()), domains, {}, leaves)
    cubes: List[Tuple[Cube, str]] = []
    for path, out in leaves:
        if out == default:
            continue  # disjoint cubes: anything not matched falls to the default
        cube = tuple(frozenset([path[i]]) if i in path else frozenset(domains[i]) for i in range(len(domains)))
        cubes.append((cube, out))
    return _merge_cubes(cubes)


# --------------------------------------------------------- decision-list strategy
def _grow(
    seed: Case,
    out: str,
    members: List[Tuple[Case, str]],
    domains: List[List[str]],
) -> Tuple[List[Set[str]], int]:
    """Greedily widen a point into the largest cube that stays pure w.r.t. ``members``."""
    n = len(domains)
    cube: List[Set[str]] = [{v} for v in seed]

    def count() -> int:
        k = 0
        for case, o in members:
            for i in range(n):
                if case[i] not in cube[i]:
                    break
            else:
                if o != out:
                    return -1
                k += 1
        return k

    cur = count()
    while True:
        best: Optional[Tuple[int, int, str]] = None
        for i in range(n):
            for v in domains[i]:
                if v in cube[i]:
                    continue
                cube[i].add(v)
                g = count()
                cube[i].discard(v)
                if g > cur and (best is None or g > best[0]):
                    best = (g, i, v)
        if best is None:
            break
        cube[best[1]].add(best[2])
        cur = best[0]
    # try to drop whole conditions (widen to the full domain) when still pure
    for i in range(n):
        if len(cube[i]) < len(domains[i]):
            old = set(cube[i])
            cube[i] = set(domains[i])
            c = count()
            if c < 0:
                cube[i] = old
            else:
                cur = c
    return cube, cur


def _decision_list_cubes(
    full: Dict[Case, str], domains: List[List[str]], default: str, max_seeds: int
) -> List[Tuple[Cube, str]]:
    n = len(domains)
    uncovered: Dict[Case, str] = dict(full)
    result: List[Tuple[Cube, str]] = []
    while True:
        targets = [c for c, o in uncovered.items() if o != default]
        if not targets:
            break
        if len(targets) > max_seeds:
            stride = math.ceil(len(targets) / max_seeds)
            seeds = targets[::stride]
        else:
            seeds = targets
        members = list(uncovered.items())
        best: Optional[Tuple[int, List[Set[str]], str]] = None
        for seed in seeds:
            cube, gain = _grow(seed, uncovered[seed], members, domains)
            if best is None or gain > best[0]:
                best = (gain, cube, uncovered[seed])
        assert best is not None
        _, cube, out = best
        result.append((tuple(frozenset(s) for s in cube), out))
        covered = [c for c in uncovered if all(c[i] in cube[i] for i in range(n))]
        for c in covered:
            del uncovered[c]
    return result


# ------------------------------------------------------------------- orchestration
def _cubes_to_rules(
    cubes: List[Tuple[Cube, str]],
    names: List[str],
    domains: List[List[str]],
    output_order: Mapping[str, int],
    sort: bool,
) -> List[Rule]:
    if sort:
        def key(item):
            cube, out = item
            size = 1
            for s in cube:
                size *= len(s)
            pos = tuple(tuple(i for i, v in enumerate(domains[j]) if v in cube[j]) for j in range(len(domains)))
            return (output_order.get(out, 99), -size, pos)

        cubes = sorted(cubes, key=key)
    rules = []
    for n, (cube, out) in enumerate(cubes, 1):
        cond = {
            names[i]: [v for v in domains[i] if v in cube[i]]
            for i in range(len(domains))
            if len(cube[i]) < len(domains[i])
        }
        rules.append(Rule(id=f"R{n}", conditions=cond, output=out))
    return rules


def compile_table(
    spec: Spec,
    approved: Union[Approved, Mapping[str, str]],
    *,
    version: str = "1",
    strategy: str = "auto",
    default_strategy: str = "safe",
    max_cases: int = 500_000,
    max_seeds: int = 64,
    verify: bool = True,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> DecisionTable:
    """Compile approved decisions into a :class:`DecisionTable`.

    strategy
        ``"auto"`` (default): ``decision_list`` when the space has at most
        ``AUTO_LIST_LIMIT`` (2,500) cases, otherwise ``tree``.
        ``"decision_list"``: ordered first-match rules found greedily; usually the
        most compact and readable, but its search is roughly quadratic in the
        number of cases. ``"tree"``: disjoint rules from an entropy-split decision
        tree plus cube merging; order-independent and fast (hundreds of thousands
        of cases in seconds).
        Neither is guaranteed to be globally minimal, but both are *exactly*
        equivalent to the approved map (checked before returning).
    default_strategy
        Which output the table's final ``else`` returns. ``"safe"`` uses the
        spec's ``safe_default`` (fail-safe, recommended); ``"majority"`` uses the
        most frequent output; ``"best"`` tries every output and keeps the fewest rules.
    verify
        Run :func:`verify_table` (including the spec's invariants) and raise
        :class:`VerificationError` on failure.
    """
    if strategy not in STRATEGIES:
        raise SpecError(f"strategy must be one of {STRATEGIES}")
    if default_strategy not in DEFAULT_STRATEGIES:
        raise SpecError(f"default_strategy must be one of {DEFAULT_STRATEGIES}")
    if spec.size > max_cases:
        raise SpaceTooLargeError(f"input space has {spec.size:,} cases (limit {max_cases:,})")
    if strategy == "auto":
        strategy = "decision_list" if spec.size <= AUTO_LIST_LIMIT else "tree"

    mapping = approved.mapping if isinstance(approved, Approved) else dict(approved)
    provenance = approved.provenance if isinstance(approved, Approved) else {}
    reviewers = approved.reviewers if isinstance(approved, Approved) else []

    names = spec.feature_names
    domains = [list(f.values) for f in spec.features]
    valid_keys = set()
    full: Dict[Case, str] = {}
    missing = 0
    for combo in itertools.product(*domains):
        key = "|".join(f"{n}={v}" for n, v in zip(names, combo))
        valid_keys.add(key)
        if key in mapping:
            out = mapping[key]
            if out not in spec.outputs:
                raise SpecError(f"approved output {out!r} for {key!r} is not a valid output")
        else:
            out = spec.safe_default
            missing += 1
        full[combo] = out
    stray = [k for k in mapping if k not in valid_keys]
    if stray:
        raise SpecError(f"approved map has {len(stray)} case(s) outside the spec's space, e.g. {stray[0]!r}")

    output_order = {o: i for i, o in enumerate(spec.output_names)}

    def build(default: str) -> List[Rule]:
        if strategy == "tree":
            cubes = _tree_cubes(full, domains, default)
            return _cubes_to_rules(cubes, names, domains, output_order, sort=True)
        cubes = _decision_list_cubes(full, domains, default, max_seeds)
        return _cubes_to_rules(cubes, names, domains, output_order, sort=False)

    counts = Counter(full.values())
    if default_strategy == "safe":
        default = spec.safe_default
        rules = build(default)
    elif default_strategy == "majority":
        top = max(counts.values())
        tied = [o for o in spec.output_names if counts.get(o, 0) == top]
        default = spec.safe_default if spec.safe_default in tied else tied[0]
        rules = build(default)
    else:  # best
        best = None
        for cand in spec.output_names:
            r = build(cand)
            rank = (len(r), 0 if cand == spec.safe_default else 1)
            if best is None or rank < best[0]:
                best = (rank, cand, r)
        _, default, rules = best  # type: ignore[misc]

    table = DecisionTable(
        name=spec.name,
        version=str(version),
        features=[f for f in spec.features],
        outputs=spec.output_names,
        safe_default=spec.safe_default,
        default_output=default,
        rules=rules,
        ordered=(strategy == "decision_list"),
        invariants=list(spec.invariants),
    )

    # Exact equivalence with the approved map (should hold by construction).
    for combo, expected in full.items():
        got = table.evaluate(dict(zip(names, combo)), validate=False).output
        if got != expected:
            raise CompilationError(f"internal error: compiled table disagrees with approved map at {combo}")

    from . import __version__

    prov_counts = dict(Counter(provenance.values())) if provenance else {}
    if missing:
        prov_counts["unmapped_to_safe_default"] = missing
    table.metadata = {
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "compiler": f"decisionforge {__version__}",
        "strategy": strategy,
        "default_strategy": default_strategy,
        "spec_hash": spec.spec_hash(),
        "cases_total": len(full),
        "cases_by_provenance": prov_counts,
        "output_distribution": dict(counts),
        "reviewers": list(reviewers),
        "rule_count": len(rules),
    }
    if extra_metadata:
        table.metadata.update(extra_metadata)

    if verify:
        report = verify_table(table, approved=mapping, max_cases=max(max_cases, spec.size))
        if not report.ok:
            from .errors import VerificationError

            raise VerificationError(report)
        table.metadata["verification"] = report.to_dict()
    return table
