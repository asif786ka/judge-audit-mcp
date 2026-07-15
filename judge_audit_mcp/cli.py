"""CI gate:  judge-audit drift v1.jsonl v2.jsonl --fail-on serious

Exits non-zero when findings meet the threshold, so judge drift becomes a red PR
instead of a wrong number in a launch doc.

The gate deliberately ignores `heuristic` findings whatever their severity. An
underpowered estimate must never fail a build — a tool that red-builds on noise
gets switched off within a week, and then it audits nothing at all. See
`models.temper_confidence`.
"""
from __future__ import annotations

import argparse
import json
import sys

from .calibrate import calibrate
from .drift import detect_drift
from .emit import emit_probe_set
from .io import LoadError
from .models import (
    CONF_HEURISTIC,
    AuditResult,
    SEV_CLEAN,
    SEV_INVALIDATING,
    SEV_MINOR,
    SEV_SERIOUS,
)
from .probes import PROBES

_ICON = {SEV_INVALIDATING: "🛑", SEV_SERIOUS: "⚠️ ", SEV_MINOR: "ℹ️ ", SEV_CLEAN: "✅"}
_LEVEL = {SEV_INVALIDATING: "error", SEV_SERIOUS: "warning", SEV_MINOR: "note",
          SEV_CLEAN: "none"}


def _text(res: AuditResult, verbose: bool) -> str:
    out = [f"judge-audit — {res.kind}", f"judge: {res.fingerprint.describe()}"]
    if res.n_items:
        out.append(f"items: {res.n_items}")
    out.append("")
    if not res.findings:
        out.append("✅ clean — no findings")
        return "\n".join(out)

    order = {SEV_INVALIDATING: 3, SEV_SERIOUS: 2, SEV_MINOR: 1, SEV_CLEAN: 0}
    for f in sorted(res.findings, key=lambda f: -order[f.severity]):
        out.append(f"{_ICON.get(f.severity, ' ')} {f.title}")
        if f.detail:
            for ln in f.detail.split("\n"):
                out.append(f"    {ln.strip()}")
        if f.effect is not None:
            out.append(f"    {f.effect_label}: {f.describe_effect()}")
        if f.confidence == CONF_HEURISTIC:
            out.append("    [heuristic — will not gate]")
        if verbose:
            for e in [e for e in f.evidence if e][:6]:
                out.append(f"      → {e}")
            if f.fix:
                out.append(f"    fix: {f.fix}")
        out.append("")
    c = res.counts()
    out.append(f"{c[SEV_INVALIDATING]} invalidating · {c[SEV_SERIOUS]} serious · "
               f"{c[SEV_MINOR]} minor · {c[SEV_CLEAN]} clean")
    for n in res.notes:
        if n:
            out.append(f"note: {n}")
    return "\n".join(out)


def _sarif(res: AuditResult) -> dict:
    """SARIF 2.1.0 — renders as inline PR annotations on GitHub."""
    rules, results = {}, []
    for f in res.findings:
        if f.severity == SEV_CLEAN:
            continue
        rules[f.id] = {
            "id": f.id,
            "name": f.title,
            "shortDescription": {"text": f.title},
            "fullDescription": {"text": f.why or f.detail[:900]},
            "help": {"text": f.fix or ""},
            "properties": {"tags": f.tags, "confidence": f.confidence},
        }
        msg = f.detail or f.title
        if f.effect is not None:
            msg += f"\n{f.effect_label}: {f.describe_effect()}"
        results.append({
            "ruleId": f.id,
            "level": _LEVEL.get(f.severity, "note"),
            "message": {"text": msg},
            "locations": [{
                "physicalLocation": {
                    "artifactLocation": {"uri": "eval"},
                    "region": {"startLine": 1},
                }
            }],
        })
    return {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [{
            "tool": {"driver": {
                "name": "judge-audit",
                "informationUri": "https://github.com/asif786ka/judge-audit-mcp",
                "rules": list(rules.values()),
            }},
            "results": results,
        }],
    }


def _emit_report(res: AuditResult, args) -> None:
    if args.format == "json":
        print(json.dumps(res.to_dict(), indent=2))
    elif args.format == "sarif":
        print(json.dumps(_sarif(res), indent=2))
    else:
        print(_text(res, args.verbose))


def _gate(res: AuditResult, fail_on: str) -> int:
    if not fail_on or fail_on == "never":
        return 0
    gating = res.gating(fail_on)
    if gating:
        print(f"\n✗ failing: {len(gating)} finding(s) at or above {fail_on}",
              file=sys.stderr)
        for f in gating:
            print(f"    {f.title}", file=sys.stderr)
        return 1
    print(f"\n✓ passing: no findings at or above {fail_on}", file=sys.stderr)
    return 0


def main(argv=None) -> int:
    # Shared flags live on a parent parser and are attached to every subcommand as
    # well as the root, so both `judge-audit --fail-on serious drift a b` and the
    # far more natural `judge-audit drift a b --fail-on serious` work. Argparse
    # only accepts root-level flags *before* the subcommand, and nobody types them
    # that way.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--format", choices=("text", "json", "sarif"), default="text")
    common.add_argument("--verbose", action="store_true")
    common.add_argument("--fail-on", default="",
                        choices=("", "never", "minor", "serious", "invalidating"),
                        help="exit non-zero on findings at or above this severity "
                             "(heuristic findings never gate)")

    p = argparse.ArgumentParser(
        prog="judge-audit",
        parents=[common],
        description="Audit the LLM judge that grades your evals. "
                    "Everyone runs a judge; nobody checks the judge.")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("drift", parents=[common],
                       help="did the judge change underneath two runs?")
    d.add_argument("run_a")
    d.add_argument("run_b")
    d.add_argument("--anchors", default="", help="comma-separated frozen control item_ids")

    c = sub.add_parser("calibrate", parents=[common],
                       help="does the judge agree with humans? (κ)")
    c.add_argument("judge")
    c.add_argument("gold")
    c.add_argument("--weights", default="auto",
                   choices=("auto", "linear", "quadratic", "none"))
    c.add_argument("--human-ceiling", type=float, default=0.0,
                   help="inter-human κ; without it we can't tell a bad judge from an "
                        "underdefined task")

    b = sub.add_parser("probe", parents=[common], help="test for a specific bias")
    b.add_argument("probe", choices=sorted(PROBES))
    b.add_argument("path")
    b.add_argument("--gold", default="")
    b.add_argument("--judge-family", default="")
    b.add_argument("--scale-min", type=float, default=0.0)
    b.add_argument("--scale-max", type=float, default=0.0)
    b.add_argument("--claimed-delta", type=float, default=0.0)

    e = sub.add_parser("emit", parents=[common],
                       help="generate a probe variant set to judge")
    e.add_argument("probe", choices=("position", "verbosity", "sycophancy"))
    e.add_argument("path")
    e.add_argument("--out", default="")
    e.add_argument("--hint", default="new_model")

    args = p.parse_args(argv)

    try:
        if args.cmd == "drift":
            ids = [s.strip() for s in args.anchors.split(",") if s.strip()] or None
            res = detect_drift(args.run_a, args.run_b, ids)
        elif args.cmd == "calibrate":
            res = calibrate(args.judge, args.gold, args.weights,
                            args.human_ceiling or None)
        elif args.cmd == "probe":
            kw: dict = {"gold_path": args.gold, "judge_family": args.judge_family}
            if args.probe == "distribution":
                kw = {"scale_min": args.scale_min or None,
                      "scale_max": args.scale_max or None,
                      "claimed_delta": args.claimed_delta or None}
            res = PROBES[args.probe](args.path, **kw)
        elif args.cmd == "emit":
            dest, n = emit_probe_set(args.path, args.probe, args.out, args.hint)
            print(f"wrote {n} record(s) to {dest}")
            print(f"score these with your judge, then: "
                  f"judge-audit probe {args.probe} {dest}")
            return 0
        else:
            p.error(f"unknown command {args.cmd}")
            return 2
    except (LoadError, FileNotFoundError, ValueError) as ex:
        print(f"error: {ex}", file=sys.stderr)
        return 2

    _emit_report(res, args)
    return _gate(res, args.fail_on)


if __name__ == "__main__":
    sys.exit(main())
