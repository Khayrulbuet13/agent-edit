"""Hiding class names from the agent, three ways:

--hide-names: every real class name becomes class_<k>.
--permute-names: every real class name becomes ANOTHER class's real name (a seeded derangement,
so no class keeps its own name); `inp`/`out` stay a bijection, same as hidden mode. The point is
to test whether the agent leans on the name rather than the data: say nothing in the prompt that
hints the names moved, so give it a `shown=` mapping already fixed (e.g. from harness.json on
replay) rather than recomputing it, since the mapping must stay the one the run actually used.

Both modes also take `secret`: extra words with no class of their own (e.g. a merged project's
parent species names, spelled out in its image keys, like cub-10's "California Gull" inside
"California_Gull_0100_41088"). Each becomes a one-way secret_<k> token -- there's nothing to map
back, since the agent never has to send one of these back in a request.

Names are matched without case and with any run of spaces, '_' or '-' between words, because
image keys and file names spell them that way. A name only matches as a whole word: letters and
digits may not touch it ('_' may, as in image keys).
"""

from __future__ import annotations

import random
import re


def derangement(n: int, seed) -> list[int]:
    """A permutation of range(n) with no fixed point (n <= 1 has none to avoid, so it's identity)."""
    if n <= 1:
        return list(range(n))
    rng = random.Random(seed)
    order = list(range(n))
    while True:
        rng.shuffle(order)
        if all(order[i] != i for i in range(n)):
            return order


class Names:
    def __init__(self, classes: list[str], secret: list[str] | None = None, permute: bool = False,
                seed=None, shown: dict | None = None):
        if not classes:
            raise ValueError("hiding class names needs the class list (backend.project_info()['classes'])")
        if shown is not None:
            self.shown = dict(shown)
        elif permute:
            if seed is None:
                raise ValueError("--permute-names needs the run's seed")
            order = derangement(len(classes), seed)
            self.shown = {c: classes[order[k]] for k, c in enumerate(classes)}
        else:
            self.shown = {c: f"class_{k}" for k, c in enumerate(classes)}
        self.real = {v: k for k, v in self.shown.items()}
        secret = sorted({w for w in (secret or []) if w not in self.shown})
        self._secret_shown = {w: f"secret_{k}" for k, w in enumerate(secret)}
        self._key = {self._norm(w): tok for w, tok in {**self.shown, **self._secret_shown}.items()}
        words = list(self.shown) + list(self._secret_shown)
        alts = [r"[\s_\-]+".join(re.escape(w) for w in re.split(r"[\s_\-]+", c.strip()))
                for c in sorted(words, key=len, reverse=True)]
        self._pat = re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(alts) + r")(?![A-Za-z0-9])", re.I) if alts else None

    @staticmethod
    def _norm(s: str) -> str:
        return re.sub(r"[\s_\-]+", " ", s.strip()).lower()

    def text(self, s: str) -> str:
        return s if self._pat is None else self._pat.sub(lambda m: self._key[self._norm(m.group(0))], s)

    def out(self, obj):
        """Copy of a result with every name in keys and strings replaced."""
        if isinstance(obj, str):
            return self.text(obj)
        if isinstance(obj, dict):
            return {(self.text(k) if isinstance(k, str) else k): self.out(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(self.out(v) for v in obj)
        return obj

    def inp(self, obj):
        """The agent's request with shown-name strings turned back into real names. Secret tokens
        have no class, so they pass through unchanged (the agent never has to echo one back)."""
        if isinstance(obj, str):
            return self.real.get(obj, obj)
        if isinstance(obj, dict):
            return {k: self.inp(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.inp(v) for v in obj]
        return obj
