---
name: class-layout
description: The order members are declared in within a class in this repo - declared API first (including __init__), then accessors/properties, then private helpers last. Load when writing a new class, adding a member to an existing one, or reviewing a class whose members look scattered.
---

# Class member layout

Members go in three sections, in this order, in **every** class:

1. **Declared API** — what callers invoke, `__init__` and other dunders included.
2. **Accessors / properties** — what callers read.
3. **Private methods and helpers** — what neither of the above is, including
   `@staticmethod` helpers.

Mark the sections with a comment so the boundary survives later edits:

```python
class Block:
    """..."""

    # 1. Declared API ---------------------------------------------------

    def __init__(self, entry):
        self._entry = entry

    def signature(self, *, deslash: bool = False):
        ...

    # 2. Accessors ------------------------------------------------------

    @property
    def hash(self):
        return self._entry.get('hash')

    # 3. Helpers --------------------------------------------------------

    @staticmethod
    def _key_from(keyby, hash, tag, version):
        ...
```

## Why this order

A reader arrives asking *what can I call*, not *how is it computed*. The
answer is at the top, and the machinery is where it can be skipped. It also
gives a diff a stable home: a new method has one obvious place to go, so
members stop accreting wherever the last edit happened to end.

## Prefer a static helper to a private method

A helper that does not need `self` is a `@staticmethod` on the class that owns
it, taking what it needs as arguments — **not** a private method on some other
class it happens to be called from. That is a namescoping decision, and it
matters most where the other class has a crowded namespace: a helper hung off
a `pandas.Series` subclass shares its attribute space with every column name in
the data.

The exception is a helper genuinely shared more broadly. Then it belongs where
its callers can all reach it, and is not private to any one of them.

## Mirroring another class's API

When a class exists to mimic part of another — a proxy, a view, a recorded
snapshot — mirror the **shape**, not merely the names: what is a property there
is a property here, and what is a method there is a method here. Code written
against one then reads the other unchanged, and `entry.paths` silently handing
back a bound method instead of a dict cannot happen.

Mirror only what the class can actually answer. A view over recorded data
should not carry the original's `build()` or `read()` just to look complete —
and must not reuse a name whose meaning differs, which is worse than omitting
it.
