"""Assign each number to its dot (one-to-one) and report what doesn't fit.

Cost = distance from the dot centre to the number's bounding box, in dot radii.
Labels are printed right next to their dot, so box distance is much more
telling than centre distance (a two-digit label's centre is far from its dot).
The Hungarian algorithm finds the globally cheapest assignment; pairs beyond a
gate distance are left unmatched rather than forced.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .detect import Dots
from .ocr import Token

CLOSE_MAX = 2.0     # close last -> first only if within this many median segments
GATE = 6.0          # max dot-to-label-box distance, in dot radii
LONG_JUMP = 4.0     # flag segments longer than this many times the median


@dataclass
class Match:
    path: list[tuple[int, np.ndarray]]          # (number, dot xy), sorted by number
    expected: list[int]                         # the sequence the puzzle should contain
    missing: list[int] = field(default_factory=list)
    duplicates: dict[int, int] = field(default_factory=dict)   # number -> times read
    unmatched_dots: list[int] = field(default_factory=list)    # indices into dots
    unmatched_tokens: list[Token] = field(default_factory=list)
    long_jumps: list[tuple[int, int]] = field(default_factory=list)
    inferred: list[int] = field(default_factory=list)          # numbers placed by elimination

    def drawing_path(self) -> list[tuple[int, np.ndarray]]:
        """The path to draw: `path`, closed with a last segment from the last
        number back to the first, when both ends were placed."""
        p = list(self.path)
        if self.closes:
            p.append(p[0])
        return p

    @property
    def closes(self) -> bool:
        """Close the outline only if both ends were placed and the last dot is
        near the first (within CLOSE_MAX x the median segment). A long closing
        line would cut across the picture instead of finishing its outline
        (picture_2: 47 -> 1 is ~5x a normal segment)."""
        if not (len(self.path) > 2 and self.expected and self.path[0][0] == self.expected[0]
                and self.path[-1][0] == self.expected[-1]):
            return False
        pts = np.array([p for _, p in self.path])
        seg = np.median(np.linalg.norm(np.diff(pts, axis=0), axis=1))
        return bool(np.linalg.norm(pts[-1] - pts[0]) <= CLOSE_MAX * seg)

    def summary(self) -> str:
        found = len(self.path)
        lines = [f"numbers placed: {found}/{len(self.expected)}"]
        if self.expected:
            ends = (len(self.path) > 2 and self.path[0][0] == self.expected[0]
                    and self.path[-1][0] == self.expected[-1])
            why = "drawn" if self.closes else \
                "not drawn (the ends are far apart)" if ends else "not drawn (an end is missing)"
            lines.append(f"closing segment {self.expected[-1]} -> {self.expected[0]}: {why}")
        step = self.expected[1] - self.expected[0] if len(self.expected) > 1 else 1
        if step != 1 or self.expected[0] != 1:
            lines[0] += f" (sequence {self.expected[0]}, {self.expected[0] + step}, ..., " \
                        f"{self.expected[-1]})"
        if self.missing:
            lines.append(f"missing: {self.missing}")
        if self.duplicates:
            lines.append("read more than once: " +
                         ", ".join(f"{v} (x{c})" for v, c in sorted(self.duplicates.items())))
        if self.inferred:
            lines.append(f"placed by elimination: {self.inferred}")
        if self.unmatched_dots:
            lines.append(f"dots without a number: {len(self.unmatched_dots)}")
        if self.unmatched_tokens:
            lines.append("numbers without a dot: " +
                         ", ".join(str(t.value) for t in self.unmatched_tokens))
        if self.long_jumps:
            lines.append("suspiciously long segments: " +
                         ", ".join(f"{a}->{b}" for a, b in self.long_jumps))
        return "\n".join(lines)


def box_distance(pts: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Distance from each point to an axis-aligned box (0 if inside)."""
    x0, y0, x1, y1 = box
    dx = np.maximum.reduce([x0 - pts[:, 0], np.zeros(len(pts)), pts[:, 0] - x1])
    dy = np.maximum.reduce([y0 - pts[:, 1], np.zeros(len(pts)), pts[:, 1] - y1])
    return np.hypot(dx, dy)


def infer_sequence(tokens: list[Token], n_dots: int, last: int | None = None) -> list[int]:
    """The numbers the puzzle should contain.

    Most puzzles count 1, 2, 3, ...; some count in steps (0, 2, 4, ... or 5, 10, ...).
    The step is the largest of 10/5/2 that divides nearly all numbers read; the
    last number is the largest one read (ignoring outliers far above the number
    of dots or labels found) unless given.
    """
    values = [t.value for t in tokens]
    step = 1
    for k in (10, 5, 2):
        if len(values) >= 10 and np.mean([v % k == 0 for v in values]) >= 0.85:
            step = k
            break
    if last is None:
        # Either count can be short (missed dots, unread labels): use the larger.
        n = max(n_dots, len(values))
        plausible = [v for v in values if v % step == 0 and v <= (n * 1.2 + 2) * step]
        last = max(plausible, default=n * step)
    first = 0 if step > 1 and 0 in values else step
    return list(range(first, last + 1, step))


BIG = 1e6


RECOVERED_PENALTY = 1.0   # dot radii: a recovered dot must be clearly closer to win


def _cost_matrix(dots: Dots, tokens: list[Token]) -> np.ndarray:
    pts, r = dots.centers, dots.radius
    owner = dots.owners()
    penalty = RECOVERED_PENALTY * (owner >= 0)
    cost = np.full((len(tokens), len(pts)), BIG)
    for i, t in enumerate(tokens):
        d = box_distance(pts, t.box) / r
        # Tie-break with centre distance so a dot between two labels goes to the closer one.
        d = d + 0.05 * np.linalg.norm(pts - t.center, axis=1) / r
        # A recovered dot was found next to one specific label: only it may take it.
        allowed = (d <= GATE) & ((owner < 0) | (owner == t.value))
        cost[i] = np.where(allowed, d + penalty, BIG)
    return cost


def _assign(cost: np.ndarray, tokens: list[Token]) -> dict[int, list[tuple[float, int, int]]]:
    """Hungarian assignment -> {number: [(cost, token idx, dot idx), ...]}."""
    by_value: dict[int, list[tuple[float, int, int]]] = {}
    if cost.size:
        for r, c in zip(*linear_sum_assignment(cost)):
            if cost[r, c] < BIG:
                by_value.setdefault(tokens[r].value, []).append((cost[r, c], r, c))
    return by_value


def _sequence_cost(p: np.ndarray, v: int, step: int, unique: dict[int, np.ndarray],
                   fallback: float) -> float:
    """How well a dot fits between the (uniquely read) dots of its two neighbours."""
    nb = [unique[k] for k in (v - step, v + step) if k in unique]
    return sum(np.linalg.norm(p - q) for q in nb) if nb else fallback


ELIM_MAX = 10        # never guess more than this many numbers by elimination
ELIM_DETOUR = 2.0    # max detour through a guessed dot, in typical segment lengths


def _eliminate(path: dict, missing: list[int], free: list[int], pts: np.ndarray,
               step: int) -> list[tuple[int, int]]:
    """Place missing numbers on leftover dots by geometry, when that is sound.

    Only when the leftover (detected) dots are about as many as the missing
    numbers, i.e. they are real dots whose labels weren't read, not junk. Each
    dot goes to the missing number whose neighbours v-step and v+step it fits
    between best (smallest detour), via a Hungarian assignment; implausible
    detours are refused. Returns (number, dot index) for each placement.
    """
    if not missing or not free or len(missing) > ELIM_MAX or len(free) > len(missing) + 1:
        return []
    seq = sorted(path)
    segs = [np.linalg.norm(path[b] - path[a]) for a, b in zip(seq, seq[1:]) if b == a + step]
    if not segs:
        return []
    seg = float(np.median(segs))
    cost = np.full((len(missing), len(free)), BIG)
    for i, v in enumerate(missing):
        nb = [path[k] for k in (v - step, v + step) if k in path]
        for j, c in enumerate(free):
            x = pts[c]
            if len(nb) == 2:
                d = np.linalg.norm(nb[0] - x) + np.linalg.norm(x - nb[1]) - np.linalg.norm(nb[0] - nb[1])
            elif len(nb) == 1:
                d = np.linalg.norm(nb[0] - x)
            else:
                continue
            if d <= ELIM_DETOUR * seg:
                cost[i, j] = d
    placed = []
    for i, j in zip(*linear_sum_assignment(cost)):
        if cost[i, j] < BIG:
            path[missing[i]] = pts[free[j]]
            placed.append((missing[i], free[j]))
    return placed


def match(dots: Dots, tokens: list[Token], last: int | None = None,
          eliminate: bool = True) -> Match:
    """`last`: the highest number in the puzzle, if known (otherwise inferred).
    `eliminate`: fill remaining gaps by geometry (see _eliminate); off while
    other evidence (re-reading labels) can still fill them."""
    pts = dots.centers
    expected = infer_sequence(tokens, len(pts), last)
    step = expected[1] - expected[0] if len(expected) > 1 else 1
    wanted = set(expected)
    rejected = [t for t in tokens if t.value not in wanted]
    tokens = [t for t in tokens if t.value in wanted]

    cost = _cost_matrix(dots, tokens)
    by_value = _assign(cost, tokens)
    duplicates = {v: len(p) for v, p in by_value.items() if len(p) > 1}
    if duplicates:
        # A number read twice is usually one good reading plus a misread of a
        # longer label (the "3" of "35"). Keep the reading whose dot lies closest
        # to its neighbours in the sequence, drop the others, and re-assign so the
        # dots the misreads had taken become free for their real labels.
        unique = {v: pts[p[0][2]] for v, p in by_value.items() if len(p) == 1}
        keep = set(range(len(tokens)))
        for v, cands in by_value.items():
            if len(cands) > 1:
                winner = min(cands, key=lambda c: _sequence_cost(pts[c[2]], v, step, unique, c[0]))
                keep -= {r for _, r, _ in cands if r != winner[1]}
        tokens = [t for i, t in enumerate(tokens) if i in keep]
        cost = _cost_matrix(dots, tokens)
        by_value = _assign(cost, tokens)
    best = {v: min(p) for v, p in by_value.items()}

    used_dots = {c for _, _, c in best.values()}
    used_tokens = {r for _, r, _ in best.values()}
    path = {v: pts[c] for v, (_, _, c) in best.items()}
    missing = [v for v in expected if v not in path]
    unmatched = [c for c in range(len(pts)) if c not in used_dots]

    free = [c for c in unmatched if dots.owners()[c] < 0]
    elim = _eliminate(path, missing, free, pts, step) if eliminate else []
    inferred = [v for v, _ in elim]
    missing = [v for v in missing if v not in inferred]
    unmatched = [c for c in unmatched if c not in {c for _, c in elim}]

    ordered = sorted(path.items())
    long_jumps = []
    seq = [(v, p) for v, p in ordered]
    lengths = [(a[0], b[0], np.linalg.norm(a[1] - b[1]))
               for a, b in zip(seq, seq[1:]) if b[0] == a[0] + step]
    if lengths:
        med = np.median([l for *_, l in lengths])
        long_jumps = [(a, b) for a, b, l in lengths if l > LONG_JUMP * med]

    return Match(
        path=ordered, expected=expected, missing=missing, duplicates=duplicates,
        unmatched_dots=unmatched,
        unmatched_tokens=[t for i, t in enumerate(tokens) if i not in used_tokens] + rejected,
        long_jumps=long_jumps, inferred=inferred,
    )


def draw_match(img: np.ndarray, dots: Dots, tokens: list[Token], m: Match) -> np.ndarray:
    vis = img.copy()
    r = int(dots.radius)
    for t in tokens:
        x0, y0, x1, y1 = t.box.astype(int)
        cv2.rectangle(vis, (x0, y0), (x1, y1), (200, 200, 200), 1)
    for c in m.unmatched_dots:
        x, y = dots.centers[c].astype(int)
        cv2.circle(vis, (x, y), 3 * r, (0, 0, 255), 2)
    for v, (x, y) in m.path:
        x, y = int(x), int(y)
        color = (0, 140, 255) if v in m.inferred else (0, 170, 0)
        cv2.circle(vis, (x, y), 2 * r, color, 2)
        cv2.putText(vis, str(v), (x + 2 * r, y - 2 * r), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (180, 0, 180), 2, cv2.LINE_AA)
    return vis
