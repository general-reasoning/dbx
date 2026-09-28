---
name: class-layout
description: The order members are declared in within a class in this repo - the Datablock/Datastack protocol and hooks first, then the rest of the declared API, then accessors/properties, then private helpers last - and the _like_this_() naming every private helper and _LikeThis_ naming every private class uses. Load when writing a new class, adding a member to an existing one, naming a private helper, or reviewing a class whose members look scattered.
---

# Class member layout

Members go in four sections, in this order, in **every** class:

1. **Protocol and hooks** — the Datablock/Datastack surface the framework
   itself calls: `__init__`, `__build__`, `__read__`, `__valid_topic__`,
   `__post_init__`, `valid`, and the rest of the dunder hooks.
2. **The rest of the declared API** — what callers invoke.
3. **Accessors / properties** — what callers read.
4. **Private methods and helpers** — what neither of the above is, including
   `@staticmethod` helpers.

Mark the sections with a comment so the boundary survives later edits:

```python
class Block(Datablock):
    """..."""

    # 1. Datablock protocol ---------------------------------------------

    def __init__(self, entry):
        self._entry = entry

    def __build__(self):
        ...

    def valid(self):
        ...

    # 2. Declared API ---------------------------------------------------

    def signature(self, *, deslash: bool = False):
        ...

    # 3. Accessors ------------------------------------------------------

    @property
    def hash(self):
        return self._entry.get('hash')

    # 4. Helpers --------------------------------------------------------

    @staticmethod
    def _key_from_(keyby, hash, tag, version):
        ...
```

## Why the protocol comes first

A Datablock subclass is read to answer *what does this block build, and is it
valid* before anything else. Those methods are the contract with dbx; every
other member exists to serve them. Burying `__build__` at the bottom — where
the training loop that implements it happened to be written last — makes a
reader scroll past the machinery to find the point.

## A name says who may call it -- and who may override it

Three spellings, three contracts:

- **`__hook__`** -- a hook: the framework calls it and a subclass implements
  it. `__build__`, `__read__`, `__block__`, `__tab__`, `__stack__`,
  `__valid_topic__`. The ONLY methods a subclass is meant to override.
- **`plain_name`** -- declared API: callers call it. A subclass may extend it
  through `super()`, but should not need to.
- **`_name_`** -- private: neither callers nor subclasses. Anything that
  feeds a block's IDENTITY (hash, key, path) belongs here: a public name is an
  invitation to override, and an override there moves every hash a subclass
  has built under. `signature_topics()` was public, and an agent reached for
  overriding it to make a specialization resolve -- it is now
  `_topics_signature_()`, and `Datablock.__init_subclass__` refuses a class
  that still defines the old name, since an override of a name nothing calls
  would be ignored in silence and change the hash.

When renaming a public member that subclasses may have overridden, make the
old name an error on definition, not a silent no-op.

## Private helpers are named `_like_this_`

Leading **and** trailing underscore on every private `def`: methods,
`@staticmethod`/`@classmethod` helpers, private `@property` accessors, and
private module-level functions. `_resume_plan_`, `_ckpt_step_`,
`_local_workdir_`, `_default_source_`.

The same for every private **class**, module-level or nested:
`_ClassOrInstance_`, `_DataSliceMeta_`, `_CallableExecutorBase_`. A class is
a member of its module's namespace exactly as a method is of its class's.

**Private means this module only.** A module-level name another dbx module
imports is not private, whatever it is called: `TopicMarkerMeta`,
`is_topicmarker`, `ShardSync` and `one_datalake` lost their underscores for
that reason. Tests do not count -- a test may reach for a private to pin it
down. And a private nothing uses is deleted, not kept "in case":
`_parse_slice_entries_` and `_extract_slice_data_` were. The trailing underscore is what
distinguishes a helper this code owns from the single-underscore names that
arrive from elsewhere.

Three things it does **not** apply to:

- **Data attributes** -- most of them. `self._entry`, `self._lightning_module`
  may keep the plain leading underscore. Two exceptions take both
  underscores: an attribute that captures an `__init__()` argument, as given,
  is `_{arg}_` -- `_tag_`, `_use_specializations_`, `_SPECIALIZATIONS_` --
  which is what `__getstate__` reads back to reproduce the call; and a
  private property, which is still a `def`, even though it is read as an
  attribute. Every private METHOD has both, without exception.
- **Nested closures.** A function defined inside a method is a local, not a
  member — it shares no namespace with anything, so there is nothing for the
  trailing underscore to disambiguate. `_atexit_sync` inside `__build__` stays
  as it is.
- **A name declared by a base class you do not control.** An override must
  spell the name exactly as the base does or it silently stops overriding: the
  framework goes on calling the base method, the subclass's version is never
  reached, and nothing errors. So check for a `super()` call or a base-class
  definition before renaming anything private.

  Within dbx both sides are ours, so the fix is to rename *both* — that is how
  `Datablock._UNSAFE_copy_topic_` and its `Still` override were converted
  together. Only a base class in another package forces the old spelling to
  stay. Renaming a base method is still a breaking change for any subclass
  outside this repo, which has to be renamed in the same breath.

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
