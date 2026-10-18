"""What an arrow between two points means: the direction of their embeddings' difference, read across the whole corpus, then named.

An arrow's two endpoints only say what differs between them, which their
labels say already. What the arrow means is its direction, and a direction
is the same anywhere in the space: it is read where else it runs.

- **Analogies.** Every pair of neighbors in the corpus is an arrow too.
  Those whose unit difference points most nearly the same way -- cosine
  with this one's -- are the same move made elsewhere: what they have in
  common is what the direction is.
- **Its axis.** The corpus projected on the direction: the points at its
  low end and at its high end are what it runs from, and toward, whoever
  its endpoints are.

That evidence, not the endpoints' texts, is what Claude is given (`prompt`),
and asked to name the direction in a line; the evidence's strength -- how
many other pairs make the same move, and how nearly -- is shown with it.
"""

from dataclasses import dataclass, field

import numpy as np

# The analogies Claude is given, at most, and the cosine below which a pair is no analogy.
ANALOGIES = 5
ANALOGY_COS = 0.5
# Pairs at least this alike count as "the same move", in the evidence shown.
SAME_COS = 0.7
# Points of each end of the axis Claude is given.
AXIS_POINTS = 5
# The longest line Claude is asked for.
NOTE_WORDS = 12


@dataclass
class Evidence:
    """Where else an arrow's direction runs: other pairs making the same move, and the ends of its axis, by their labels."""

    analogies: list = field(default_factory=list)  # (from, to, cosine), the most alike first
    low: list = field(default_factory=list)  # the axis's low end: what the direction runs from
    high: list = field(default_factory=list)  # its high end: what it runs toward

    def same(self):
        """How many other pairs make the same move, and the nearest's cosine."""
        alike = [c for _, _, c in self.analogies if c >= SAME_COS]
        return len(alike), max((c for _, _, c in self.analogies), default=0.0)


def unit_rows(m):
    m = np.asarray(m, dtype=np.float64)
    return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)


def neighbor_moves(units, neighbors):
    """Every arrow the corpus's neighbors make: the pairs ``[(a, b)]``, and their unit differences, a row each (zero for none)."""
    pairs = [(a, int(b)) for a in range(len(units)) for b in neighbors[a]]
    if not pairs:
        return pairs, np.zeros((0, units.shape[1]))
    return pairs, unit_rows(np.array([units[b] - units[a] for a, b in pairs]))


def evidence(units, i, j, pairs, moves, labels):
    """Arrow *i* -> *j*'s `Evidence`, over unit embeddings *units* and the corpus's *moves*; None where *i* and *j* are one embedding."""
    d = units[j] - units[i]
    norm = float(np.linalg.norm(d))
    if norm < 1e-9:
        return None
    d = d / norm
    out = Evidence()
    cos = moves @ d if len(moves) else np.zeros(0)
    seen = {(labels[i], labels[j])}
    for k in np.argsort(-cos):
        a, b = pairs[k]
        if cos[k] < ANALOGY_COS or len(out.analogies) == ANALOGIES:
            break
        if {a, b} == {i, j} or (labels[a], labels[b]) in seen:
            continue  # itself, or a pair the corpus holds again -- the same pipeline run again
        seen.add((labels[a], labels[b]))
        out.analogies.append((labels[a], labels[b], float(cos[k])))
    along = (units - units.mean(axis=0)) @ d
    for end, order in ((out.low, np.argsort(along)), (out.high, np.argsort(-along))):
        for k in order:
            if labels[k] not in end:
                end.append(labels[k])
            if len(end) == AXIS_POINTS:
                break
    return out


def prompt(arrows):
    """What Claude is asked, of *arrows*: ``[(from label, to label, Evidence)]``."""
    lines = [
        "Each item below is an arrow between two neighboring points of an embedding: the direction of the "
        "difference of their embeddings. Its two endpoints only say what differs between them. What the arrow "
        "means is its direction, read where else it runs in the corpus:",
        "- the same move elsewhere: other pairs of points whose difference points nearly the same way, with the "
        "cosine of the two directions;",
        "- its axis: the points of the whole corpus lying furthest toward each end of the direction.",
        f"For each arrow, name the direction itself in one line of at most {NOTE_WORDS} words: what changes along "
        "it, as all of that evidence has in common. Do not describe the two endpoints, and do not list the "
        "evidence. Answer with a JSON array of strings, one per arrow, in order, and nothing else.",
        "",
    ]
    for n, (here, there, ev) in enumerate(arrows, 1):
        lines.append(f'{n}. from "{here}" to "{there}"')
        lines.append("   the same move elsewhere:" if ev.analogies else "   the same move elsewhere: none")
        lines += [f'     "{a}" -> "{b}" (cos {c:.2f})' for a, b, c in ev.analogies]
        lines.append("   its axis runs from: " + "; ".join(f'"{x}"' for x in ev.low))
        lines.append("   toward: " + "; ".join(f'"{x}"' for x in ev.high))
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
