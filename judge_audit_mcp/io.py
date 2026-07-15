"""Loading judge outputs and human labels.

Deliberately, aggressively tolerant. Everyone's eval log has a different shape:
LangSmith calls it `key`, Braintrust calls it `input`, the intern's notebook
calls it `q_id`, and half of them nest the score under `outputs.score`. A tool
that demands a schema audits nobody's judge, because the migration is never worth
it for a tool you haven't been convinced by yet.

So: JSONL, JSON array, or CSV; field names guessed from a synonym table;
arbitrary nesting flattened with dots. Guessing is *reported*, never silent —
`LoadReport.describe()` tells you which column it read as the score, because a
tool that silently reads the wrong column is worse than one that fails.

One firm rule in all this softness: `item_id` is required and never invented. If
records can't be paired across two runs, every comparison downstream is between
different populations, and inventing IDs by row order would fabricate exactly
that. Row order is not an identifier. Better to fail loudly here than to report a
confident number about a join that never happened.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Optional

from .models import GoldLabel, JudgeFingerprint, JudgeRecord

# --- field synonyms ----------------------------------------------------------
# Ordered by priority: the first hit wins.

_ID_KEYS = ("item_id", "id", "example_id", "sample_id", "case_id", "qid", "q_id",
            "key", "uid", "index", "example", "question_id", "row_id", "test_id")
_SCORE_KEYS = ("score", "judge_score", "rating", "grade", "value", "points",
               "judge_rating", "overall", "overall_score", "quality", "result_score")
_CHOICE_KEYS = ("choice", "winner", "preferred", "preference", "verdict_choice",
                "better", "selected", "pick")
_LABEL_KEYS = ("label", "verdict", "judgment", "judgement", "decision", "class",
               "category", "outcome", "passed", "pass_fail")
_TEXT_KEYS = ("output_text", "output", "response", "completion", "answer", "text",
              "candidate", "generation", "prediction")
_LEN_KEYS = ("output_len", "length", "n_tokens", "tokens", "token_count", "n_chars")
_FAMILY_KEYS = ("output_family", "family", "model_family", "candidate_model",
                "output_model", "generator", "model_under_test", "system")
_POSITION_KEYS = ("position", "slot", "order", "arm")
_VARIANT_KEYS = ("variant", "condition", "probe_variant", "treatment", "arm_variant")
_HUMAN_KEYS = ("human_score", "gold_score", "human_rating", "gold", "human",
               "reference_score", "ground_truth", "target", "expected")
_NRATERS_KEYS = ("n_raters", "num_raters", "raters", "n_annotators")
_RATER_AGREE_KEYS = ("rater_agreement", "inter_rater_kappa", "human_kappa",
                     "iaa", "inter_annotator_agreement")

# Judge-config fields, for fingerprinting.
_MODEL_KEYS = ("judge_model", "model", "evaluator_model", "grader_model", "judge")
_PROVIDER_KEYS = ("provider", "judge_provider", "api", "vendor")
_TEMP_KEYS = ("temperature", "judge_temperature", "temp")
_SCALE_KEYS = ("scale", "score_scale", "rubric_scale", "range")
_PROMPT_KEYS = ("judge_prompt", "prompt", "system_prompt", "instructions", "template")
_RUBRIC_KEYS = ("rubric", "criteria", "rubric_text", "grading_criteria")
_PROMPT_HASH_KEYS = ("prompt_hash", "judge_prompt_hash", "template_hash")
_RUBRIC_HASH_KEYS = ("rubric_hash", "criteria_hash")
_VERSION_KEYS = ("judge_version", "version", "snapshot", "revision")

_TRUTHY = {"pass", "true", "yes", "1", "correct", "good", "win", "accept"}
_FALSY = {"fail", "false", "no", "0", "incorrect", "bad", "lose", "reject"}


class LoadError(Exception):
    """Raised when a file cannot be read as judge records at all."""


class LoadReport:
    """What we guessed, so the user can check it.

    Exists because the failure mode of tolerant parsing is confident nonsense:
    read `latency_ms` as the score column and this server will happily audit your
    judge's response times and report κ.
    """

    def __init__(self) -> None:
        self.path: str = ""
        self.n_rows: int = 0
        self.n_dropped: int = 0
        self.mapping: dict[str, str] = {}
        self.warnings: list[str] = []

    def describe(self) -> str:
        bits = [f"{self.n_rows} record(s) from {self.path}"]
        if self.mapping:
            bits.append("read as: " + ", ".join(f"{v} → {k}" for k, v in self.mapping.items()))
        if self.n_dropped:
            bits.append(f"{self.n_dropped} row(s) dropped (no usable id or verdict)")
        for w in self.warnings:
            bits.append(f"! {w}")
        return "\n".join(bits)


# --- raw reading -------------------------------------------------------------

def _flatten(d: Any, prefix: str = "") -> dict:
    """Flatten nested dicts to dotted keys, keeping the leaf name as an alias.

    The alias is what makes `{"outputs": {"score": 4}}` findable as `score`
    without the caller knowing the nesting. Shallow keys win ties, since a
    top-level `score` is more likely the one you meant than a buried one.
    """
    out: dict = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
            continue
        out[key] = v
        if prefix and k not in out:
            out[k] = v
    return out


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise LoadError(f"No such file: {path}")
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        raise LoadError(f"Empty file: {path}")

    suffix = path.suffix.lower()
    if suffix in (".csv", ".tsv"):
        delim = "\t" if suffix == ".tsv" else ","
        return list(csv.DictReader(text.splitlines(), delimiter=delim))

    # A whole-file JSON document: an array of records, or an object wrapping one.
    # Tried before JSONL, and by *parsing* rather than by sniffing for newlines —
    # pretty-printed JSON is multi-line and would otherwise fall through to the
    # JSONL branch and fail on every line of it.
    if text[0] in "[{":
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None                     # Probably JSONL. Fall through.
        if isinstance(data, list):
            return [r for r in data if isinstance(r, dict)]
        if isinstance(data, dict):
            for k in ("records", "results", "rows", "data", "examples", "runs",
                      "items", "evaluations", "outputs"):
                if isinstance(data.get(k), list):
                    return [r for r in data[k] if isinstance(r, dict)]
            return [data]

    # JSONL — the common case. Tolerate blank lines and trailing junk.
    rows, bad = [], 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    if not rows:
        raise LoadError(f"No JSON objects found in {path} ({bad} unparseable line(s))")
    return rows


def _pick(row: dict, keys: Iterable[str]) -> tuple[Optional[str], Any]:
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return k, row[k]
    lower = {str(k).lower(): k for k in row}
    for k in keys:
        real = lower.get(k)
        if real is not None and row[real] not in (None, ""):
            return real, row[real]
    return None, None


def _as_float(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        s = str(v).strip().lower()
        if s in _TRUTHY:
            return 1.0
        if s in _FALSY:
            return 0.0
        return None


def _as_label(v: Any) -> Optional[str]:
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        return "pass" if v else "fail"
    return str(v).strip()


# --- fingerprints ------------------------------------------------------------

def fingerprint_from_row(row: dict) -> JudgeFingerprint:
    """Recover a fingerprint from whatever the log happened to record.

    Prompt/rubric are hashed if present in full, or taken as-is if the harness
    already stored a hash. Most logs have neither, which is the finding.
    """
    _, model = _pick(row, _MODEL_KEYS)
    _, provider = _pick(row, _PROVIDER_KEYS)
    _, temp = _pick(row, _TEMP_KEYS)
    _, scale = _pick(row, _SCALE_KEYS)
    _, version = _pick(row, _VERSION_KEYS)
    _, prompt = _pick(row, _PROMPT_KEYS)
    _, rubric = _pick(row, _RUBRIC_KEYS)
    _, phash = _pick(row, _PROMPT_HASH_KEYS)
    _, rhash = _pick(row, _RUBRIC_HASH_KEYS)

    fp = JudgeFingerprint.from_config(
        model=str(model or ""),
        prompt=str(prompt or ""),
        rubric=str(rubric or ""),
        scale=str(scale or ""),
        temperature=_as_float(temp),
        provider=str(provider or ""),
        version=str(version or ""),
    )
    # An explicitly logged hash beats one we computed.
    if phash or rhash:
        fp = JudgeFingerprint(
            model=fp.model,
            prompt_hash=str(phash) if phash else fp.prompt_hash,
            rubric_hash=str(rhash) if rhash else fp.rubric_hash,
            scale=fp.scale,
            temperature=fp.temperature,
            provider=fp.provider,
            version=fp.version,
        )
    return fp


def consensus_fingerprint(records: list[JudgeRecord]) -> tuple[JudgeFingerprint, list[str]]:
    """The fingerprint of a run, plus warnings if it isn't singular.

    A run whose records disagree about the judge is *already* broken — that's a
    within-run drift, and it means the mean score for that run is an average over
    two different instruments. Rare, and catastrophic when it happens (someone's
    retry logic silently fell back to a different model on rate-limit).
    """
    if not records:
        return JudgeFingerprint(), []
    seen: dict[str, tuple[JudgeFingerprint, int]] = {}
    for r in records:
        h = r.fingerprint.hash()
        fp, c = seen.get(h, (r.fingerprint, 0))
        seen[h] = (fp, c + 1)
    if len(seen) == 1:
        return next(iter(seen.values()))[0], []
    ranked = sorted(seen.values(), key=lambda t: -t[1])
    warn = [f"Run contains {len(seen)} distinct judge fingerprints — the records "
            f"in this file were not all judged by the same instrument. "
            + "; ".join(f"{fp.describe()} ({c} rec)" for fp, c in ranked[:3])]
    return ranked[0][0], warn


# --- public loaders ----------------------------------------------------------

def load_judge_records(path: str) -> tuple[list[JudgeRecord], LoadReport]:
    p = Path(path).expanduser()
    rows = _read_rows(p)
    rep = LoadReport()
    rep.path = str(p)

    out: list[JudgeRecord] = []
    for i, raw in enumerate(rows):
        row = _flatten(raw)
        k_id, vid = _pick(row, _ID_KEYS)
        k_sc, vsc = _pick(row, _SCORE_KEYS)
        k_ch, vch = _pick(row, _CHOICE_KEYS)
        k_lb, vlb = _pick(row, _LABEL_KEYS)

        if vid is None or (vsc is None and vch is None and vlb is None):
            rep.n_dropped += 1
            continue

        k_tx, vtx = _pick(row, _TEXT_KEYS)
        k_ln, vln = _pick(row, _LEN_KEYS)
        k_fm, vfm = _pick(row, _FAMILY_KEYS)
        k_ps, vps = _pick(row, _POSITION_KEYS)
        k_vr, vvr = _pick(row, _VARIANT_KEYS)

        for field_name, k in (("item_id", k_id), ("score", k_sc), ("choice", k_ch),
                              ("label", k_lb), ("output_text", k_tx), ("output_len", k_ln),
                              ("output_family", k_fm), ("position", k_ps), ("variant", k_vr)):
            if k and field_name not in rep.mapping:
                rep.mapping[field_name] = k

        out.append(JudgeRecord(
            item_id=str(vid),
            score=_as_float(vsc),
            choice=_as_label(vch),
            label=_as_label(vlb),
            output_family=str(vfm or "").strip(),
            output_text=str(vtx or ""),
            output_len=int(_as_float(vln)) if _as_float(vln) is not None else None,
            position=_as_label(vps),
            variant=str(vvr or "").strip(),
            fingerprint=fingerprint_from_row(row),
            meta={},
        ))

    if not out:
        raise LoadError(
            f"{p}: found {len(rows)} row(s) but none had both an id and a verdict. "
            f"Expected an id field (one of: {', '.join(_ID_KEYS[:5])}...) and a "
            f"score/choice/label field. Columns present: {', '.join(list(rows[0])[:12])}"
        )

    fp, warns = consensus_fingerprint(out)
    rep.warnings.extend(warns)
    if fp.is_empty():
        rep.warnings.append(
            "No judge fingerprint in this file (no model / prompt / rubric / scale "
            "recorded). Scores from this run cannot be proven comparable to any "
            "other run — not now, and not retrospectively.")
    rep.n_rows = len(out)
    return out, rep


def load_gold_labels(path: str) -> tuple[list[GoldLabel], LoadReport]:
    p = Path(path).expanduser()
    rows = _read_rows(p)
    rep = LoadReport()
    rep.path = str(p)

    out: list[GoldLabel] = []
    for raw in rows:
        row = _flatten(raw)
        k_id, vid = _pick(row, _ID_KEYS)
        # Human files often name the column `human_score`; fall back to plain `score`.
        k_sc, vsc = _pick(row, _HUMAN_KEYS)
        if vsc is None:
            k_sc, vsc = _pick(row, _SCORE_KEYS)
        k_ch, vch = _pick(row, _CHOICE_KEYS)
        k_lb, vlb = _pick(row, _LABEL_KEYS)
        if vid is None or (vsc is None and vch is None and vlb is None):
            rep.n_dropped += 1
            continue

        _, vnr = _pick(row, _NRATERS_KEYS)
        _, vra = _pick(row, _RATER_AGREE_KEYS)
        for field_name, k in (("item_id", k_id), ("score", k_sc),
                              ("choice", k_ch), ("label", k_lb)):
            if k and field_name not in rep.mapping:
                rep.mapping[field_name] = k

        nr = _as_float(vnr)
        out.append(GoldLabel(
            item_id=str(vid),
            score=_as_float(vsc),
            choice=_as_label(vch),
            label=_as_label(vlb),
            n_raters=int(nr) if nr else 1,
            rater_agreement=_as_float(vra),
            meta={},
        ))

    if not out:
        raise LoadError(f"{p}: no rows with both an id and a human verdict.")
    rep.n_rows = len(out)
    return out, rep


# --- pairing -----------------------------------------------------------------

def pair_by_item(
    judge: list[JudgeRecord],
    gold: list[GoldLabel],
) -> tuple[list[tuple[JudgeRecord, GoldLabel]], list[str]]:
    """Inner join on item_id. Returns pairs and warnings about what didn't join.

    Duplicate judge records for one id are dropped rather than silently
    de-duplicated to the first: a repeated id usually means either a retry or a
    pairwise layout (two outputs per item), and both would corrupt a pointwise
    κ if we guessed. Loud and unhelpful beats quiet and wrong.
    """
    warns: list[str] = []
    gmap: dict[str, GoldLabel] = {}
    dupes_g = 0
    for g in gold:
        if g.item_id in gmap:
            dupes_g += 1
            continue
        gmap[g.item_id] = g

    seen: dict[str, int] = {}
    for r in judge:
        seen[r.item_id] = seen.get(r.item_id, 0) + 1
    dupes_j = [k for k, v in seen.items() if v > 1]

    pairs = [(r, gmap[r.item_id]) for r in judge
             if r.item_id in gmap and seen[r.item_id] == 1]

    unmatched_j = len({r.item_id for r in judge}) - len({r.item_id for r, _ in pairs})
    unmatched_g = len(gmap) - len(pairs)
    if dupes_j:
        warns.append(f"{len(dupes_j)} item_id(s) appear more than once in the judge file "
                     f"and were excluded (e.g. {', '.join(dupes_j[:3])}). If this is a "
                     f"pairwise eval, use bias_probe rather than calibrate_judge.")
    if dupes_g:
        warns.append(f"{dupes_g} duplicate id(s) in the human file; first kept.")
    if unmatched_j:
        warns.append(f"{unmatched_j} judged item(s) have no human label.")
    if unmatched_g:
        warns.append(f"{unmatched_g} human-labelled item(s) were never judged.")
    return pairs, warns


def pair_runs(
    a: list[JudgeRecord],
    b: list[JudgeRecord],
) -> tuple[list[tuple[JudgeRecord, JudgeRecord]], list[str]]:
    """Inner join two runs on item_id — the identical-item subset.

    This subset is the whole trick behind drift detection. Comparing mean(run_b)
    to mean(run_a) confounds three things at once: the system changed, the eval
    set changed, and the judge changed. Restricting to items present and
    unchanged in both runs holds the eval set fixed, which is what lets you
    attribute the delta.
    """
    warns: list[str] = []
    amap: dict[str, JudgeRecord] = {}
    for r in a:
        amap.setdefault(r.item_id, r)
    pairs = [(amap[r.item_id], r) for r in b if r.item_id in amap]
    only_a = len(amap) - len(pairs)
    only_b = len({r.item_id for r in b}) - len(pairs)
    if only_a or only_b:
        warns.append(f"Eval set changed between runs: {only_a} item(s) only in A, "
                     f"{only_b} only in B. Drift is measured on the {len(pairs)} "
                     f"item(s) common to both.")
    return pairs, warns
