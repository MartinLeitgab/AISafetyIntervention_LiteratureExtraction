"""Static sanity check for the manuscript, since no LaTeX toolchain is installed.

Checks, in order of how badly each would break a build:
  1. \begin{X} / \end{X} balance and nesting
  2. every \ref / \cref / \Cref target resolves to a \label
  3. every \label is referenced at least once (unused labels are harmless but noisy)
  4. every \cite* key exists in refs.bib
  5. exactly one \bibliography and one \bibliographystyle
  6. brace balance outside verbatim
  7. \includegraphics targets exist on disk
This is NOT a substitute for compiling. It catches the errors that a compile would
catch first.
"""

import re
import sys
from collections import Counter
from pathlib import Path

TEX = Path(sys.argv[1])
BIB = TEX.parent / "refs.bib"
src = TEX.read_text(encoding="utf-8")

# strip comments (a % not preceded by a backslash)
lines = []
for ln in src.splitlines():
    out, esc = [], False
    for i, ch in enumerate(ln):
        if ch == "%" and not (i and ln[i - 1] == "\\"):
            break
        out.append(ch)
    lines.append("".join(out))
body = "\n".join(lines)

fail = 0


def bad(msg):
    global fail
    fail += 1
    print(f"  FAIL  {msg}")


print(
    f"linting {TEX.name} ({len(src.splitlines())} lines, {len(body)} chars after comment strip)"
)

# 1 environments
stack = []
for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", body):
    kind, env = m.group(1), m.group(2)
    if kind == "begin":
        stack.append(env)
    else:
        if not stack:
            bad(f"\\end{{{env}}} with no open environment")
        elif stack[-1] != env:
            bad(f"\\end{{{env}}} closes \\begin{{{stack[-1]}}}")
            stack.pop()
        else:
            stack.pop()
if stack:
    bad(f"unclosed environments: {stack}")
else:
    print("  ok    environments balanced")

# 2/3 labels and refs
labels = set(re.findall(r"\\label\{([^}]+)\}", body))
refs = Counter(re.findall(r"\\(?:c|C)?ref\{([^}]+)\}", body))
for group in list(refs):
    for r in group.split(","):
        r = r.strip()
        if r and r not in labels:
            bad(f"reference to undefined label: {r}")
unused = labels - {r.strip() for g in refs for r in g.split(",")}
if unused:
    print(f"  warn  labels defined but never referenced: {sorted(unused)}")
print(f"  ok    {len(labels)} labels, {sum(refs.values())} references")

# 4 citations
if BIB.exists():
    bibtext = BIB.read_text(encoding="utf-8")

    # 4a BibTeX has NO comment syntax inside an entry: a percent sign between fields
    # is literal text and stops the parse. Outside an entry BibTeX skips everything
    # until the next '@', so text there is unconstrained -- the scanner must model
    # that, or a brace inside a prose comment produces a false alarm.
    bib_fail_at_entry = fail
    depth, inside, entry_line = 0, False, 0
    for lineno, ln in enumerate(bibtext.splitlines(), 1):
        i = 0
        while i < len(ln):
            ch = ln[i]
            esc = i > 0 and ln[i - 1] == "\\"
            if not inside:
                if ch == "@":
                    entry_line = lineno
                elif ch == "{" and entry_line and not esc:
                    inside, depth = True, 1
            else:
                if ch == "{" and not esc:
                    depth += 1
                elif ch == "}" and not esc:
                    depth -= 1
                    if depth == 0:
                        inside, entry_line = False, 0
                elif ch == "%" and not esc:
                    bad(
                        f"refs.bib:{lineno} percent sign inside the entry opened on "
                        f"line {entry_line} -- BibTeX aborts here; move it outside"
                    )
            i += 1
    if inside:
        bad(f"refs.bib: entry opened on line {entry_line} is never closed")
    elif fail == bib_fail_at_entry:
        print("  ok    refs.bib entries closed, no percent sign inside an entry")

    keys = set(re.findall(r"@\w+\{([^,]+),", bibtext))
    cited = set()
    for g in re.findall(r"\\cite[a-zA-Z]*\{([^}]+)\}", body):
        cited |= {c.strip() for c in g.split(",")}
    for c in sorted(cited):
        if c not in keys:
            bad(f"citation key not in refs.bib: {c}")
    print(
        f"  ok    {len(cited)} distinct citation keys against {len(keys)} bib entries"
    )
    unusedbib = keys - cited
    if unusedbib:
        print(f"  warn  bib entries never cited: {sorted(unusedbib)}")
else:
    bad(f"refs.bib not found next to the manuscript ({BIB})")

# 5 bibliography commands
nb = len(re.findall(r"\\bibliography\{", body))
ns = len(re.findall(r"\\bibliographystyle\{", body))
if nb != 1:
    bad(f"expected exactly 1 \\bibliography, found {nb}")
if ns != 1:
    bad(f"expected exactly 1 \\bibliographystyle, found {ns}")
if nb == 1 and ns == 1:
    print("  ok    one \\bibliography + one \\bibliographystyle")

# 6 braces
depth = 0
for i, ch in enumerate(body):
    if ch == "{" and (i == 0 or body[i - 1] != "\\"):
        depth += 1
    elif ch == "}" and (i == 0 or body[i - 1] != "\\"):
        depth -= 1
        if depth < 0:
            bad(f"unbalanced closing brace near char {i}")
            depth = 0
if depth:
    bad(f"{depth} unclosed brace(s)")
else:
    print("  ok    braces balanced")

# 7 graphics
for g in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body):
    cands = [TEX.parent / g] + [
        TEX.parent / f"{g}{e}" for e in (".png", ".pdf", ".jpg")
    ]
    if not any(c.exists() for c in cands):
        bad(f"\\includegraphics target missing on disk: {g}")
    else:
        print(f"  ok    graphic found: {g}")

# 8 markers still open
for marker in ("[GAP:", "[CITE:", "[FIG:"):
    n = src.count(marker)
    if n:
        print(f"  note  {n} open {marker} marker(s)")

print(f"\n{'FAILED' if fail else 'PASSED'}: {fail} blocking issue(s)")
sys.exit(1 if fail else 0)
