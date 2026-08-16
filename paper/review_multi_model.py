#!/usr/bin/env python3
"""Send Paper A (PDF) + the NeurIPS scoring guidance to three frontier models and
collect independent reviews.

Six jobs: {Claude Opus 5, GPT-5.6 Sol, Gemini 3.1 Pro} x {conference, workshop}.
All six run concurrently; each streams its response to disk as it arrives, so a
mid-flight kill loses nothing that already landed.

WHAT IS SENT: the paper PDF, the scoring-guidance text, and the prompt below.
Nothing else. No CLAUDE.md, no repo files, no environment or identity metadata.
Keep it that way -- do not add "helpful" context to the prompt.

Usage
-----
    export REVIEW_PAPER_PDF=/abs/path/to/paper.pdf
    export REVIEW_ANTHROPIC_ENV=/abs/path/to/.env   # the file holding ANTHROPIC_API_KEY
    python -u review_multi_model.py --smoke     # 1 cheap job, proves the plumbing
    python -u review_multi_model.py             # all 6 jobs

Both variables are required and fail fast if unset. They are environment-supplied
rather than hardcoded because this repository is public and the paths are
machine-specific. The OpenAI and Gemini key files sit inside this repo's own tree
and are resolved relative to it.

Receipts land in paper/reviews_<UTC-date>/:
    <model>_<mode>.md          the review text
    <model>_<mode>.meta.json   model id, token usage, wall-clock, finish reason
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import datetime as dt
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Inputs. Every path is absolute and checked before any model call is made.
# --------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
GUIDANCE = HERE / "neurips-scoringguidance.txt"

# This repository is public, so no machine-specific absolute path is hardcoded here.
# The two that are not derivable from the repo layout come from the environment:
#
#   REVIEW_PAPER_PDF     absolute path to the compiled paper PDF
#   REVIEW_ANTHROPIC_ENV absolute path to a .env file holding ANTHROPIC_API_KEY
#
# Both fail fast and name themselves if unset. The other two key files sit inside
# this repo's own tree, so they stay relative.
_PDF_ENV = "REVIEW_PAPER_PDF"
_ANTHROPIC_ENV = "REVIEW_ANTHROPIC_ENV"

PAPER_PDF = Path(os.environ[_PDF_ENV]) if os.environ.get(_PDF_ENV) else None

# API keys are read from .env files at runtime; never logged, never written to an
# output file.
KEY_FILES = {
    "anthropic": (
        Path(os.environ[_ANTHROPIC_ENV]) if os.environ.get(_ANTHROPIC_ENV) else None,
        "ANTHROPIC_API_KEY",
    ),
    "openai": (HERE.parent / "graph_analysis" / ".env", "openai_api_key"),
    "gemini": (HERE.parent.parent / ".env", "gemini_api_key"),
}

# On all three providers the output cap counts reasoning/thinking tokens, not just
# visible text. The smoke run proved this: an 800-token cap yielded 29 visible
# tokens and stop=MAX_TOKENS, the rest having gone to thinking. 64k is the ceiling
# all three models accept (Gemini outputTokenLimit 65536, Opus 5 and GPT-5.6 Sol
# 64k+), and leaves room for a long review after thinking is paid for.
MAX_OUTPUT_TOKENS = 64_000
SMOKE_OUTPUT_TOKENS = 800


# --------------------------------------------------------------------------
# The prompt. Identical across all three models so the reviews are comparable.
# --------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert NeurIPS reviewer. Adopt that persona for this review."
)

PROMPT_CONFERENCE = """\
Review the attached paper against the attached NeurIPS scoring guidance, and give \
the scores that guidance asks for.

In addition, flag:

1. Writing style that is unscientific or unacademic -- passages that read as \
non-human or LLM-generated, too informal, or sloppy.

2. Passages that attribute outsized importance ("critical", "essential", and \
similar) to aspects that are not the paper's main findings and are not \
load-bearing for its claims.

3. Content reporting unsuccessful designs or failures that carry no load-bearing \
insight for the paper's findings or key messages -- "sausage-making" from the \
analysis effort that does not belong in the final paper. Name each one and say \
what to cut.
"""

PROMPT_WORKSHOP_SUFFIX = """\

Review this for acceptance at a WORKSHOP, not at a main conference: apply a lower \
significance bar -- early, discussable, honestly-reported work is in scope -- and \
score accordingly. Everything above is otherwise unchanged.
"""

MODES = {
    "conference": PROMPT_CONFERENCE,
    "workshop": PROMPT_CONFERENCE + PROMPT_WORKSHOP_SUFFIX,
}


# --------------------------------------------------------------------------
# Fail-fast input loading
# --------------------------------------------------------------------------


def read_key(provider: str) -> str:
    path, var = KEY_FILES[provider]
    if path is None:
        raise SystemExit(
            f"FATAL: {provider} key file location not set.\n"
            f"  set {_ANTHROPIC_ENV} to the absolute path of a .env file "
            f"containing {var}.\n"
            f"  This script does NOT search for it and does NOT fall back to "
            f"another provider."
        )
    if not path.is_file():
        raise SystemExit(
            f"FATAL: {provider} key file not found.\n"
            f"  expected: {path}\n"
            f"  expected variable: {var}\n"
            f"  This script does NOT fall back to environment variables or to a "
            f"different provider -- fix the path or the file."
        )
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if name.strip().lower() == var.lower():
            value = value.strip().strip('"').strip("'")
            if value:
                return value
    raise SystemExit(
        f"FATAL: variable {var} not found (or empty) in {path}.\n"
        f"  This script does NOT fall back to another key source."
    )


def load_inputs() -> tuple[bytes, str]:
    if PAPER_PDF is None:
        raise SystemExit(
            f"FATAL: paper PDF location not set.\n"
            f"  set {_PDF_ENV} to the absolute path of the compiled paper PDF.\n"
            f"  This script does NOT search for it and does NOT compile LaTeX."
        )
    if not PAPER_PDF.is_file():
        raise SystemExit(
            f"FATAL: paper PDF not found.\n"
            f"  expected: {PAPER_PDF}\n"
            f"  produced by: compiling paperA_altstyle.tex on Overleaf and "
            f"downloading the PDF to that path.\n"
            f"  This script does NOT compile LaTeX and does NOT fall back to the "
            f".tex source."
        )
    if not GUIDANCE.is_file():
        raise SystemExit(f"FATAL: scoring guidance not found.\n  expected: {GUIDANCE}")
    return PAPER_PDF.read_bytes(), GUIDANCE.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Job plumbing
# --------------------------------------------------------------------------

PRINT_LOCK = threading.Lock()


def log(job: str, msg: str) -> None:
    with PRINT_LOCK:
        stamp = dt.datetime.now().strftime("%H:%M:%S")
        print(f"[{stamp}] {job:<28} {msg}", flush=True)


@dataclass
class Job:
    provider: str
    model: str
    mode: str
    label: str
    out_md: Path
    out_meta: Path
    max_tokens: int
    meta: dict = field(default_factory=dict)


class Sink:
    """Append-as-you-go writer. A killed run keeps whatever already arrived."""

    def __init__(self, path: Path, job: str):
        self.path = path
        self.job = job
        self.chars = 0
        self.last_report = 0.0
        self.fh = path.open("w", encoding="utf-8", newline="\n")

    def write(self, text: str) -> None:
        if not text:
            return
        self.fh.write(text)
        self.fh.flush()
        self.chars += len(text)
        now = time.monotonic()
        if now - self.last_report >= 30:
            self.last_report = now
            log(self.job, f"streaming... {self.chars:,} chars")

    def close(self) -> None:
        self.fh.close()


def user_text(guidance: str, mode: str) -> str:
    return (
        "NeurIPS scoring guidance follows between the markers.\n\n"
        "=== BEGIN NEURIPS SCORING GUIDANCE ===\n"
        f"{guidance}\n"
        "=== END NEURIPS SCORING GUIDANCE ===\n\n"
        f"{MODES[mode]}"
    )


# --------------------------------------------------------------------------
# Per-provider calls. Each streams and returns a usage dict.
# --------------------------------------------------------------------------


def run_anthropic(job: Job, pdf: bytes, guidance: str, sink: Sink) -> dict:
    import anthropic

    client = anthropic.Anthropic(api_key=read_key("anthropic"))
    # Thinking and effort are deliberately omitted: on claude-opus-5 thinking is
    # on (adaptive) by default and effort defaults to high, so omitting them is
    # equivalent to setting them and avoids SDK-version typing gaps.
    with client.messages.stream(
        model=job.model,
        max_tokens=job.max_tokens,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": base64.standard_b64encode(pdf).decode("ascii"),
                        },
                    },
                    {"type": "text", "text": user_text(guidance, job.mode)},
                ],
            }
        ],
    ) as stream:
        for text in stream.text_stream:
            sink.write(text)
        final = stream.get_final_message()

    if final.stop_reason == "refusal":
        raise RuntimeError(
            f"{job.label}: model refused (stop_details="
            f"{getattr(final, 'stop_details', None)}); no review produced."
        )
    return {
        "stop_reason": final.stop_reason,
        "input_tokens": final.usage.input_tokens,
        "output_tokens": final.usage.output_tokens,
    }


def run_openai(job: Job, pdf: bytes, guidance: str, sink: Sink) -> dict:
    from openai import OpenAI

    client = OpenAI(api_key=read_key("openai"))
    b64 = base64.standard_b64encode(pdf).decode("ascii")
    stream = client.responses.create(
        model=job.model,
        instructions=SYSTEM_PROMPT,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "filename": PAPER_PDF.name,
                        "file_data": f"data:application/pdf;base64,{b64}",
                    },
                    {"type": "input_text", "text": user_text(guidance, job.mode)},
                ],
            }
        ],
        reasoning={"effort": "high"},
        max_output_tokens=job.max_tokens,
        stream=True,
    )
    usage = {}
    status = None
    for event in stream:
        etype = getattr(event, "type", "")
        if etype == "response.output_text.delta":
            sink.write(event.delta)
        elif etype in ("response.completed", "response.incomplete", "response.failed"):
            resp = event.response
            status = getattr(resp, "status", etype)
            u = getattr(resp, "usage", None)
            if u is not None:
                usage = {
                    "input_tokens": getattr(u, "input_tokens", None),
                    "output_tokens": getattr(u, "output_tokens", None),
                }
        elif etype == "error":
            raise RuntimeError(f"{job.label}: stream error: {event}")
    return {"stop_reason": status, **usage}


def run_gemini(job: Job, pdf: bytes, guidance: str, sink: Sink) -> dict:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=read_key("gemini"))
    stream = client.models.generate_content_stream(
        model=job.model,
        contents=[
            types.Part.from_bytes(data=pdf, mime_type="application/pdf"),
            types.Part.from_text(text=user_text(guidance, job.mode)),
        ],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=job.max_tokens,
        ),
    )
    usage = {}
    finish = None
    for chunk in stream:
        if chunk.text:
            sink.write(chunk.text)
        if chunk.candidates and chunk.candidates[0].finish_reason:
            finish = str(chunk.candidates[0].finish_reason)
        if chunk.usage_metadata:
            usage = {
                "input_tokens": chunk.usage_metadata.prompt_token_count,
                "output_tokens": chunk.usage_metadata.candidates_token_count,
            }
    return {"stop_reason": finish, **usage}


RUNNERS = {"anthropic": run_anthropic, "openai": run_openai, "gemini": run_gemini}


def execute(job: Job, pdf: bytes, guidance: str) -> Job:
    log(job.label, f"start  model={job.model} max_out={job.max_tokens:,}")
    started = time.monotonic()
    sink = Sink(job.out_md, job.label)
    try:
        usage = RUNNERS[job.provider](job, pdf, guidance, sink)
    finally:
        sink.close()
    elapsed = time.monotonic() - started
    job.meta = {
        "provider": job.provider,
        "model": job.model,
        "mode": job.mode,
        "elapsed_sec": round(elapsed, 1),
        "response_chars": sink.chars,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        **usage,
    }
    job.out_meta.write_text(json.dumps(job.meta, indent=2), encoding="utf-8")
    log(
        job.label,
        f"done   {sink.chars:,} chars in {elapsed / 60:.1f} min "
        f"(out_tokens={usage.get('output_tokens')}, stop={usage.get('stop_reason')})",
    )
    return job


# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="one cheap Gemini job with a small output cap, to prove the plumbing "
        "and measure tokens/sec before the full run",
    )
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()

    pdf, guidance = load_inputs()
    outdir = args.outdir or (
        HERE / f"reviews_{dt.datetime.now(dt.timezone.utc).date().isoformat()}"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    specs = [
        ("anthropic", "claude-opus-5", "opus5"),
        ("openai", "gpt-5.6-sol", "gpt56sol"),
        ("gemini", "gemini-3.1-pro-preview", "gemini31pro"),
    ]
    if args.smoke:
        specs = [specs[2]]
        modes = ["conference"]
        cap = SMOKE_OUTPUT_TOKENS
    else:
        modes = list(MODES)
        cap = MAX_OUTPUT_TOKENS

    jobs = [
        Job(
            provider=provider,
            model=model,
            mode=mode,
            label=f"{short}/{mode}",
            out_md=outdir / f"{short}_{mode}.md",
            out_meta=outdir / f"{short}_{mode}.meta.json",
            max_tokens=cap,
        )
        for provider, model, short in specs
        for mode in modes
    ]

    print(
        f"PDF {PAPER_PDF.stat().st_size / 1e6:.2f} MB | guidance "
        f"{GUIDANCE.stat().st_size / 1024:.1f} KB | {len(jobs)} job(s) | out -> {outdir}",
        flush=True,
    )

    failures: list[tuple[str, BaseException]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {pool.submit(execute, j, pdf, guidance): j for j in jobs}
        for fut in concurrent.futures.as_completed(futures):
            job = futures[fut]
            try:
                fut.result()
            except BaseException as exc:  # noqa: BLE001 - reported, then re-raised
                failures.append((job.label, exc))
                log(job.label, f"FAILED {type(exc).__name__}: {exc}")

    print("\n=== summary ===", flush=True)
    truncated = []
    for job in jobs:
        if job.meta:
            stop = str(job.meta.get("stop_reason") or "")
            # A review cut off at the cap is not a review. Surface it rather than
            # letting a truncated file read as a finished one.
            flag = ""
            if "max_tokens" in stop.lower() or "incomplete" in stop.lower():
                flag = "  <-- TRUNCATED AT CAP, do not read as complete"
                truncated.append(job.label)
            print(
                f"  {job.label:<24} {job.meta['response_chars']:>7,} chars  "
                f"{job.meta['elapsed_sec'] / 60:>5.1f} min  stop={stop}{flag}"
            )
        else:
            print(f"  {job.label:<24} FAILED")
    if truncated:
        print(
            f"\n{len(truncated)} job(s) hit the {MAX_OUTPUT_TOKENS:,}-token output cap: "
            f"{', '.join(truncated)}. Re-run those with a higher cap.",
            file=sys.stderr,
        )

    if failures:
        print(f"\n{len(failures)} of {len(jobs)} job(s) failed:", file=sys.stderr)
        for label, exc in failures:
            print(f"  {label}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
