"""
Heuristic fact extraction: turns raw conversation text into typed claims.

This is the free, always-on tier. It is deliberately regex-and-syscall only --
no model, no embeddings -- because it runs inside a hook on every prompt the
user submits. When a connected agent volunteers structured facts through
`omnimem_save_session`, those arrive with higher authority and outrank these.

Two of the sources here are not heuristics at all but ground truth, which is
why they punch above their weight:

* **Stack** comes from reading the manifests on disk.
* **Map** entities are filtered through os.path.exists(), so one syscall
  eliminates every hallucinated file path.
"""

import json
import os
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from omnimem.core import facts as facts_mod

Claim = Tuple[str, str]  # (kind, text)

MAX_PER_TURN = 8
MAX_ENTITIES_PER_TURN = 4

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`\"'\[(])")
_CHECKBOX = re.compile(r"^\s*[-*]\s*\[( |x|X)\]\s*(.+)$")
_CODEISH = re.compile(r"^(?:\s{4,}|\t|```|import |from |def |class |\}|\{|<|\$ |#!)")
_HEADING = re.compile(r"^\s*#{1,6}\s")

_CUES: Sequence[Tuple[str, re.Pattern]] = (
    # Order is the classifier. Gotchas first because they usually *contain* a
    # decision cue ("turns out you have to use X"), and the warning is the part
    # worth keeping.
    ("gotcha", re.compile(
        r"\b(turns out|the (?:problem|issue|bug) (?:was|is)|beware|careful|caveat|gotcha|"
        r"note that|fails? because|(?:doesn't|does not|didn't) work|broke\w* because|"
        r"caused by|watch out|silently|footgun|surprising|the catch|pitfall|"
        r"root cause|because of a bug)\b", re.I)),
    # Hard constraints outrank goals: "must be free" is a boundary, not an aim.
    ("constraint", re.compile(
        r"\b(must not|must|cannot|can'?t|never|do not|don'?t|without spending|"
        r"zero cost|no budget|free tier|only ever|at most|no more than|"
        r"has to be|is not allowed|not allowed to|hard requirement)\b", re.I)),
    ("open_question", re.compile(
        r"\b(not sure|unclear|unresolved|open question|still deciding|tbd|"
        r"to be decided|wondering (?:if|whether)|question is|undecided)\b", re.I)),
    ("goal", re.compile(
        r"\b(i want|i'?d like|i'?m trying to|i am trying to|we want|the goal is|"
        r"goal:|objective|aim(?:ing)? to|so that i can|trying to build|"
        r"i'?m building|we'?re building|working on|end state|ultimately)\b", re.I)),
    ("decision", re.compile(
        r"\b(we'?ll use|let'?s use|decided to|decision:|chose|chosen|going with|"
        r"switch(?:ed|ing)? to|instead of|rather than|opt(?:ed)? for|settled on|"
        r"we use|i'?ll use|will use|in favour of|in favor of|replaced? \w+ with)\b", re.I)),
    ("constraint", re.compile(
        r"\b(needs? to be|should not|shouldn'?t|requirement|required to|"
        r"limit(?:ed)? to|budget|constraint)\b", re.I)),
    ("convention", re.compile(
        r"\b(always|by convention|convention is|naming|style guide|we prefer|"
        r"prefer(?:red)? to|we follow|standard is|format is|keep it)\b", re.I)),
    ("task", re.compile(
        r"^\s*(?:todo|to-?do)\b|\b(next step|next up|then (?:we|i)|need to add|"
        r"still need|remaining|left to do|follow-?up)\b", re.I)),
)

# The user is authoritative about intent; the assistant is authoritative about
# what was actually built. Cross-assigned kinds are dropped rather than
# down-weighted, because a model narrating "your goal is X" is just an echo.
#
# Gotchas are allowed from *both* sides on purpose. Users report them constantly
# ("turns out the trigger doesn't fire"), and a gotcha is the highest-value line
# in a brief -- restricting them to the assistant threw away most of them.
_ROLE_KINDS = {
    "user": {"goal", "constraint", "task", "open_question", "decision",
             "convention", "gotcha"},
    "assistant": {"decision", "gotcha", "convention", "constraint", "task", "entity", "stack"},
}

_PATH_RE = re.compile(
    r"(?<![\w/])((?:[\w.\-]+/)*[\w.\-]+\."
    r"(?:py|js|mjs|cjs|ts|tsx|jsx|json|md|toml|ya?ml|sql|html|css|scss|go|rs|"
    r"java|kt|rb|sh|zsh|env|cfg|ini|txt|lock|proto|graphql))(?![\w/])"
)


def _units(text: str) -> List[str]:
    """Split a turn into candidate claim-sized units."""
    out: List[str] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.rstrip()
        if not line.strip() or _CODEISH.match(line) or _HEADING.match(line):
            continue
        box = _CHECKBOX.match(line)
        if box:
            if box.group(1) == " ":       # unticked only; done items are noise
                out.append("TODO " + box.group(2).strip())
            continue
        line = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", line).strip()
        for sentence in _SENTENCE_SPLIT.split(line):
            sentence = sentence.strip()
            if sentence:
                out.append(sentence)
    return out


def _classify(unit: str) -> Optional[str]:
    for kind, pattern in _CUES:
        if pattern.search(unit):
            return kind
    return None


def _usable(unit: str) -> bool:
    if not (18 <= len(unit) <= 400):
        return False
    # Two separate floors. Raw words guard against fragments; content tokens
    # guard against pure filler. Gating on content tokens alone was wrong --
    # "It must not cost anything to run" folds to three content tokens and is
    # exactly the kind of hard constraint the brief exists to carry.
    if len(re.findall(r"\b[\w'-]+\b", unit)) < 4:
        return False
    if len(facts_mod.token_set(unit)) < 3:
        return False
    # A line that is mostly punctuation or symbols is markup, not a claim.
    letters = sum(ch.isalpha() or ch.isspace() for ch in unit)
    return letters / len(unit) >= 0.6


def extract_claims(text: str, role: str = "user") -> List[Claim]:
    """Typed claims from one conversational turn."""
    allowed = _ROLE_KINDS.get(role, _ROLE_KINDS["user"])
    claims: List[Claim] = []
    seen: set = set()
    for unit in _units(text):
        if not _usable(unit):
            continue
        kind = "task" if unit.startswith("TODO ") else _classify(unit)
        if kind is None or kind not in allowed:
            continue
        cleaned = facts_mod.clean_text(unit[5:] if unit.startswith("TODO ") else unit)
        key = facts_mod.canonical(cleaned)
        if not cleaned or key in seen:
            continue
        seen.add(key)
        claims.append((kind, cleaned))
        if len(claims) >= MAX_PER_TURN:
            break
    return claims


# --------------------------------------------------------------------------- map

def extract_entities(text: str, root: Optional[str]) -> List[Claim]:
    """
    File-path claims, verified against the filesystem.

    An unverified path is worse than no path: it sends the next session looking
    for a file that never existed. os.path.exists() is the whole filter.

    The gloss is built per *unit* rather than by scanning for the nearest "."
    in the raw text -- a filename contains dots, so the naive scan cut
    "omnimem/core/render.py" in half and glossed one path with a fragment of
    another.

    A gloss is only attached when the sentence names exactly one path. With two
    or more, removing them leaves a gutted fragment ("Implemented the store in
    and the renderer in") that reads as broken and says less than the bare path.
    """
    if not root or not os.path.isdir(root):
        return []
    out: List[Claim] = []
    seen: set = set()
    for unit in _units(text):
        matches = list(_PATH_RE.finditer(unit))
        if not matches:
            continue
        gloss = ""
        if len(matches) == 1:
            gloss = _PATH_RE.sub("", unit)
            gloss = re.sub(r"\s{2,}", " ", gloss).strip(" ,-–—:.")
            # Drop a dangling preposition left behind by the removal.
            gloss = re.sub(
                r"\s+(?:in|into|to|at|for|from|on|of|with|and|the|a|an)$", "", gloss, flags=re.I
            )
            gloss = facts_mod.clean_text(gloss)
        for match in matches:
            rel = match.group(1).lstrip("./")
            if rel in seen or not os.path.exists(os.path.join(root, rel)):
                continue
            seen.add(rel)
            out.append(("entity", f"{rel} — {gloss[:90]}" if len(gloss) >= 12 else rel))
            if len(out) >= MAX_ENTITIES_PER_TURN:
                return out
    return out


# --------------------------------------------------------------------------- stack

def _read(path: str, limit: int = 60000) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read(limit)
    except OSError:
        return ""


# Packages worth naming in a brief. A pip freeze is mostly transitive deps, and
# "first 12 alphabetically" surfaces `annotated-types` while hiding `fastapi` --
# so known frameworks are promoted ahead of declaration order. Not exhaustive by
# design; anything unrecognised still shows up, just after these.
_NOTABLE = frozenset("""
fastapi flask django starlette uvicorn gunicorn hypercorn litestar sanic tornado
pydantic sqlalchemy alembic psycopg psycopg2 asyncpg pymongo redis celery
fastmcp mcp anthropic openai langchain llama-index transformers torch tensorflow
numpy pandas scipy scikit-learn polars duckdb pyarrow
pytest hypothesis tox nox ruff black mypy flake8 pylint coverage
httpx requests aiohttp websockets typer click rich textual
react next vue svelte angular vite webpack esbuild rollup tailwindcss
express fastify koa nest prisma drizzle typeorm mongoose
typescript eslint prettier jest vitest playwright cypress storybook
""".split())

_PINNED_RE = re.compile(r"==")


def _rank_deps(names: List[str]) -> List[str]:
    notable = [n for n in names if n.lower() in _NOTABLE]
    rest = [n for n in names if n.lower() not in _NOTABLE]
    return notable + rest


def _requirements(root: str) -> Optional[str]:
    body = _read(os.path.join(root, "requirements.txt"))
    if not body:
        return None
    names: List[str] = []
    pinned = 0
    for line in body.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        if _PINNED_RE.search(line):
            pinned += 1
        name = re.split(r"[=<>!~\[;\s]", line, 1)[0].strip()
        if name and name not in names:
            names.append(name)
    if not names:
        return None
    ranked = _rank_deps(names)
    # A fully-pinned file with a long tail is a `pip freeze`, where declaration
    # order carries no intent -- say how many there are instead of implying the
    # first few were chosen.
    if len(names) > 25 and pinned >= len(names) * 0.9:
        return (f"Python: {len(names)} pinned deps (pip freeze); notable: "
                + ", ".join(ranked[:8]))
    return "Python deps: " + ", ".join(ranked[:12])


def _package_json(root: str) -> Optional[str]:
    body = _read(os.path.join(root, "package.json"))
    if not body:
        return None
    try:
        data = json.loads(body)
    except ValueError:
        return None
    deps = list((data.get("dependencies") or {}).keys())
    deps += list((data.get("devDependencies") or {}).keys())
    deps = _rank_deps(deps)
    label = data.get("name") or "node"
    return f"Node project '{label}' deps: " + ", ".join(deps[:12]) if deps else \
        f"Node project '{label}'"


def _pyproject(root: str) -> Optional[str]:
    body = _read(os.path.join(root, "pyproject.toml"))
    if not body:
        return None
    name = re.search(r'(?m)^\s*name\s*=\s*["\']([^"\']+)', body)
    block = re.search(r"(?s)dependencies\s*=\s*\[(.*?)\]", body)
    deps: List[str] = []
    if block:
        for item in re.findall(r'["\']([^"\']+)["\']', block.group(1)):
            dep = re.split(r"[=<>!~\[;\s]", item, 1)[0].strip()
            if dep and dep not in deps:
                deps.append(dep)
    label = f"Python project '{name.group(1)}'" if name else "Python project (pyproject.toml)"
    return f"{label} deps: {', '.join(deps[:12])}" if deps else label


def _dockerfile(root: str) -> Optional[str]:
    body = _read(os.path.join(root, "Dockerfile"))
    if not body:
        return None
    bases = list(dict.fromkeys(re.findall(r"(?mi)^\s*FROM\s+(\S+)", body)))
    return "Docker base: " + ", ".join(bases[:3]) if bases else None


def _gomod(root: str) -> Optional[str]:
    body = _read(os.path.join(root, "go.mod"))
    if not body:
        return None
    module = re.search(r"(?m)^module\s+(\S+)", body)
    version = re.search(r"(?m)^go\s+(\S+)", body)
    bits = ["Go module " + module.group(1) if module else "Go module"]
    if version:
        bits.append(f"(go {version.group(1)})")
    return " ".join(bits)


_SIMPLE_MARKERS = {
    "Cargo.toml": "Rust crate (Cargo)",
    "Gemfile": "Ruby project (Bundler)",
    "pom.xml": "Java project (Maven)",
    "build.gradle": "Java/Kotlin project (Gradle)",
    "composer.json": "PHP project (Composer)",
    "pubspec.yaml": "Dart/Flutter project",
    "Makefile": "Has a Makefile",
    "docker-compose.yml": "Uses docker-compose",
}


def scan_stack(root: Optional[str]) -> List[Claim]:
    """
    Stack facts read off the manifests. Ground truth, no NLP.

    Emitted as one aggregate claim per manifest rather than one per dependency:
    a fact per package would swamp the brief and tell the next session nothing
    it couldn't get from `cat requirements.txt`.
    """
    if not root or not os.path.isdir(root):
        return []
    out: List[Claim] = []
    for fn in (_requirements, _package_json, _pyproject, _gomod, _dockerfile):
        try:
            value = fn(root)
        except Exception:
            value = None
        if value:
            out.append(("stack", facts_mod.clean_text(value)))
    for filename, label in _SIMPLE_MARKERS.items():
        if os.path.exists(os.path.join(root, filename)):
            out.append(("stack", label))
    return out


def guess_title(text: str, fallback: str = "") -> str:
    for line in (text or "").splitlines():
        line = line.strip(" #*-")
        if 8 <= len(line) <= 70:
            return line
    return fallback


# --------------------------------------------------------------------------- gating

FLUFF_PATTERNS = (
    re.compile(r"^\s*(hi|hello|hey|greetings|good morning|good evening|good afternoon)\b.*$", re.I),
    re.compile(r"^\s*(thanks|thank you|thx|ty|ok|okay|got it|cool|sounds good|great|"
               r"awesome|sure|yes|no|yep|nope|nice|perfect)\s*[.!?]*$", re.I),
    re.compile(r"^\s*(can you help me|how are you|what'?s up|bye|goodbye)\s*[.!?]*$", re.I),
)

TECHNICAL_KEYWORDS = frozenset("""
api database sql sqlite fts5 bug fix feature config architecture server endpoint
ui react python javascript typescript function class async await test docker deploy
git error exception model schema auth token key route cache queue index migration
""".split())


def is_important(text: str) -> bool:
    """Importance gate: drop greetings and content-free acknowledgements."""
    if not text or not text.strip():
        return False
    clean = text.strip()
    for pattern in FLUFF_PATTERNS:
        if pattern.match(clean):
            return False
    if len(clean) < 15:
        words = set(re.findall(r"\b\w+\b", clean.lower()))
        if not words & TECHNICAL_KEYWORDS:
            return False
    return True


def turn_tags(text: str, limit: int = 8) -> List[str]:
    """Cheap keyword tags for the FTS/notes layer."""
    counts: Dict[str, int] = {}
    for token in facts_mod.tokens(text):
        if len(token) >= 3:
            counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [t for t, _ in ranked[:limit]] or ["general"]
