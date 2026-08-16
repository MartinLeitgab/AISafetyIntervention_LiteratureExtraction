"""Plumbing test for streaming_claude_call.

Tiny prompt -> verify the streaming wrapper:
  1. Launches `claude -p --output-format stream-json` correctly via Popen
  2. Receives stream-json events (parses each line as JSON)
  3. Extracts text deltas from content_block_delta > text_delta events
  4. Writes them to partial file as they arrive (line-buffered, flushed)
  5. Sentinel validation works in streaming_call_with_validation
  6. Max-plan billing (no ANTHROPIC_API_KEY leak)

Cost estimate: ~30k tokens (CLI auto-memory preamble dominates the tiny call).
~+5pp session at Max 5x tier.
"""

import sys
from pathlib import Path

# Load the module under test
sys.path.insert(0, str(Path(__file__).parent))
import phase2_step4_phase2_doublet_llm_grouping as mod

partial = Path("logfiles/plumbing_test_partial.txt")
partial.parent.mkdir(parents=True, exist_ok=True)

sentinel_token = "plumbtest42"
end_marker = f"END_SENTINEL_{sentinel_token}"

prompt = (
    f'Output ONLY the literal JSON `{{"hello": "world", "n": 42}}` '
    f"immediately followed by the sentinel `{end_marker}`. Do NOT add any "
    f"explanation, preamble, markdown fences, or extra whitespace. The first "
    f"character of your output must be `{{` and the last characters must be "
    f"`{end_marker}`."
)

print("=" * 60, flush=True)
print("PLUMBING TEST: streaming_call_with_validation", flush=True)
print("=" * 60, flush=True)
print(f"partial file: {partial}", flush=True)
print(f"end_marker:   {end_marker}", flush=True)
print(f"prompt len:   {len(prompt)} chars", flush=True)
print(flush=True)

json_part, dur, attempts, err = mod.streaming_call_with_validation(
    prompt, sentinel_token, "plumb", partial, max_retries=0
)

print(flush=True)
print("=" * 60, flush=True)
print("RESULT", flush=True)
print("=" * 60, flush=True)
print(f"attempts:   {attempts}", flush=True)
print(f"duration:   {dur:.1f}s", flush=True)
print(f"error:      {err}", flush=True)
print(f"json_part:  {json_part!r}", flush=True)

if json_part:
    import json as _json

    try:
        parsed = _json.loads(json_part)
        print(f"PARSED JSON: {parsed}", flush=True)
        if parsed == {"hello": "world", "n": 42}:
            print("PLUMBING TEST: PASS", flush=True)
            sys.exit(0)
        else:
            print(
                "PLUMBING TEST: PARTIAL (text streamed + parsed, but content mismatch)",
                flush=True,
            )
            sys.exit(1)
    except Exception as e:
        print(f"PLUMBING TEST: JSON parse failed: {e}", flush=True)
        sys.exit(2)
else:
    print("PLUMBING TEST: FAIL (no json_part returned)", flush=True)
    print(f"  inspect partial file at {partial}", flush=True)
    if partial.exists():
        partial_content = partial.read_text(encoding="utf-8", errors="replace")
        print(
            f"  partial content ({len(partial_content)} chars): {partial_content[:500]!r}",
            flush=True,
        )
    sys.exit(3)
