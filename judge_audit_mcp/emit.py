"""Generate probe variant sets.

The offline probes need variants — swapped orders, padded outputs, hinted
prompts. This module builds the file of things to judge; the user runs their own
judge over it and feeds the results back. That round trip is what keeps the core
of this server free of API keys, costs, and nondeterminism, while still letting
it audit a judge it cannot call.

The padding text below is deliberately dull. Filler that adds *information* would
make the padded answer genuinely better and destroy the probe's identification —
the whole claim rests on the manipulation being quality-preserving. So the filler
restates, hedges, and pads out; it never adds a fact. If you swap in your own,
keep that property or the verbosity probe stops meaning anything.
"""
from __future__ import annotations

import json
from pathlib import Path

from .io import load_judge_records
from .models import JudgeRecord

# Content-free filler. Restates and hedges; asserts nothing new.
FILLER_PREFIX = (
    "It's worth noting at the outset that this is a question that benefits from "
    "careful consideration, and there are a number of angles from which one might "
    "approach it. With that context in mind, here is the response:\n\n"
)
FILLER_SUFFIX = (
    "\n\nTo summarise the above: the key points have been laid out, and taken "
    "together they address the question as posed. It is of course worth bearing in "
    "mind that individual circumstances vary, and there may be additional nuances "
    "depending on the specifics of the situation. Overall, the considerations "
    "described above should provide a useful basis for thinking about this."
)

# Content-free hints. Each says something about the asker, nothing about the answer.
HINTS = {
    "new_model": "Note: this response is from our new model, which we believe is an improvement.",
    "production": "Note: this is the response from our production system.",
    "team_favourite": "Note: our team thinks this response is the stronger one.",
    "expensive": "Note: this response came from our most capable (and most expensive) model.",
}


def pad(text: str, mode: str = "both") -> str:
    if mode == "prefix":
        return FILLER_PREFIX + text
    if mode == "suffix":
        return text + FILLER_SUFFIX
    return FILLER_PREFIX + text + FILLER_SUFFIX


def emit_probe_set(path: str, probe: str, out_path: str = "",
                   hint: str = "new_model", pad_mode: str = "both") -> tuple[str, int]:
    """Read a judge run, write the variant set to judge next.

    Output rows carry `variant`, and every row keeps its original `item_id` —
    that pairing is what the probes join on. `score`/`choice` are left null: they
    are what your judge is being asked to fill in.
    """
    recs, _ = load_judge_records(path)
    out = Path(out_path).expanduser() if out_path else Path(path).with_suffix(f".{probe}.jsonl")
    rows: list[dict] = []

    for r in recs:
        base = {"item_id": r.item_id, "output_family": r.output_family}

        if probe == "position":
            # Emit both orders explicitly, so the file is self-contained and the
            # original run's order can't silently disagree with what we assumed.
            rows.append({**base, "variant": "", "position": "A",
                         "output_text": r.output_text, "choice": None,
                         "_instruction": "Judge with outputs in their original order."})
            rows.append({**base, "variant": "swapped", "position": "B",
                         "output_text": r.output_text, "choice": None,
                         "_instruction": "Judge with the two outputs presented in the "
                                         "opposite order. Record the slot you pick "
                                         "(A/B) as seen in THIS presentation."})
        elif probe == "verbosity":
            if not r.output_text:
                continue
            rows.append({**base, "variant": "", "output_text": r.output_text,
                         "output_len": len(r.output_text), "score": None})
            padded = pad(r.output_text, pad_mode)
            rows.append({**base, "variant": "padded", "output_text": padded,
                         "output_len": len(padded), "score": None,
                         "_instruction": "Same answer, padded with content-free filler."})
        elif probe == "sycophancy":
            rows.append({**base, "variant": "", "output_text": r.output_text,
                         "score": None, "_hint": ""})
            rows.append({**base, "variant": "hinted", "output_text": r.output_text,
                         "score": None, "_hint": HINTS.get(hint, HINTS["new_model"]),
                         "_instruction": "Prepend _hint to the judge prompt. Change "
                                         "nothing else."})
        else:
            raise ValueError(
                f"emit_probe_set does not generate variants for {probe!r}. "
                f"'distribution' and 'length_confound' need no variants — run them "
                f"directly on your existing log. 'self_preference' needs human labels, "
                f"not variants.")

    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return str(out), len(rows)
