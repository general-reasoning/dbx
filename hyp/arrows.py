"""What an arrow between two points means: their embeddings' difference, decoded against the corpus's phrases, then said in a line.

An embedding has no decoder: no text comes back out of a vector. A
difference vector ``ê_there - ê_here`` can still be read by what it points
toward. The corpus's own embed texts are cut into phrases -- a line each,
a long line at its sentences -- every phrase is embedded with the corpus's
embedder, and the arrow is read as the phrases most aligned with it (where
moving there takes you) and the most opposed (what moving there leaves
behind). The phrases are the vocabulary the arrow is spelled in: a phrase
can score high without being in either text, which is what makes this a
reading of the vector rather than of the texts.

With `dbxpipelines.LabelFirstEmbedder` the label's trigram block is part of
every vector, phrases' included: between two points of one label it
cancels, and the arrow reads as their rests' difference; between labels it
counts, and phrases sharing the target label's trigrams rank high.

The phrases alone read as a list of what one text has and the other lacks.
What is shown is a line Claude writes from them (`prompt`, `parse`): the
projection is what the arrow is read from, and Claude says what it means.
"""

import re

import numpy as np

# The phrases embedded, at most: the ones in the most texts first.
PHRASE_LIMIT = 4000
# A phrase is a line, or a sentence of a long one, this long or less.
PHRASE_CHARS = 120
# Phrases each arrow is read from, either way: what Claude is given to say what it means.
ARROW_PHRASES = 10
# The longest line Claude is asked for.
NOTE_WORDS = 12

_PUNCT_ = re.compile(r"""['"{}\[\]()\\]""")


def phrases(texts, limit=PHRASE_LIMIT):
    """The phrases of *texts* to decode arrows against: distinct, the ones in the most texts first, at most *limit*.

    A phrase in every text is dropped: it is no direction. Quotes and
    brackets are stripped, and the backslashes escaping them, so a spec's
    dict lines read as ``key: value``.
    """
    seen = {}
    for text in texts:
        for phrase in set(_cut_(text)):
            seen[phrase] = seen.get(phrase, 0) + 1
    n = len(texts)
    ranked = sorted((p for p, c in seen.items() if c < n or n == 1), key=lambda p: (-seen[p], p))
    return ranked[:limit]


def decode(here, there, phrase_vectors, k=ARROW_PHRASES):
    """The arrow from *here* to *there*, vectors: indices and scores of the *k* phrases most toward it, and the *k* most away.

    Scores are cosines between ``ê_there - ê_here`` and each phrase's
    vector less the phrases' mean, in [-1, 1]. Every embedding shares a large
    common direction; a phrase's own vector measures mostly that, and its
    distance from the mean is what it says that the others do not. Two equal
    vectors have no arrow: both lists empty.
    """
    d = _unit_(there) - _unit_(here)
    norm = float(np.linalg.norm(d))
    if norm < 1e-9 or not len(phrase_vectors):
        return [], []
    p = _unit_rows_(phrase_vectors)
    scores = _unit_rows_(p - p.mean(axis=0)) @ (d / norm)
    k = min(k, len(scores))
    order = np.argsort(-scores)
    toward = [(int(i), float(scores[i])) for i in order[:k]]
    away = [(int(i), float(scores[i])) for i in order[::-1][:k]]
    return toward, away


def prompt(arrows):
    """What Claude is asked, of *arrows*: ``[(from label, to label, toward, away)]``, phrases with their scores."""
    lines = [
        "Each item below is an arrow between two neighboring points of an embedding: from one text to another. "
        "Under it are the phrases most aligned with the difference of their embeddings -- where moving along the "
        "arrow leads (+) -- and the most opposed -- what it leaves behind (-), with their cosine scores; the order "
        "of the scores matters, not their size.",
        f"For each arrow, say in one short line, at most {NOTE_WORDS} words, what moving along it means: the gist "
        "of the change, not a list of the phrases. Answer with a JSON array of strings, one per arrow, in order, "
        "and nothing else.",
        "",
    ]
    for n, (here, there, toward, away) in enumerate(arrows, 1):
        lines.append(f'{n}. from "{here}" to "{there}"')
        lines += [f"   + {p} ({s:+.2f})" for p, s in toward] + [f"   - {p} ({s:+.2f})" for p, s in away]
    return "\n".join(lines)


def parse(reply, n):
    """*reply*'s *n* lines, one per arrow; None where it holds no JSON array of *n* strings."""
    import json

    a, b = reply.find("["), reply.rfind("]")
    try:
        got = json.loads(reply[a : b + 1]) if a >= 0 and b > a else None
    except ValueError:
        return None
    if not isinstance(got, list) or len(got) != n:
        return None
    return [" ".join(str(g).split()) for g in got]


def _cut_(text):
    for line in str(text).split("\n"):
        line = " ".join(_PUNCT_.sub("", line).split()).strip(" ,;")
        parts = [line] if len(line) <= PHRASE_CHARS else re.split(r"(?<=[.;])\s+", line)
        for p in parts:
            p = p.strip(" ,;")
            if 3 <= len(p) <= PHRASE_CHARS and re.search(r"[A-Za-z]", p):
                yield p


def _unit_(v):
    v = np.asarray(v, dtype=np.float32)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _unit_rows_(m):
    m = np.asarray(m, dtype=np.float32)
    return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)
