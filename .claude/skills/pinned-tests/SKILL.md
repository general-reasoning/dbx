---
name: pinned-tests
description: What `@pytest.mark.pinned` means in this repo - a failing pinned test is a regression, so the CODE changes, never the test. Load before editing any test to make it pass, and when deciding whether a new test should be pinned.
---

# Pinned tests

`@pytest.mark.pinned` marks a test that states an **invariant** rather than the
shape the code currently has.

The suite holds both kinds, and they call for opposite responses:

| | states | when it fails |
|---|---|---|
| ordinary | the shape the code has now | the test may be what to update |
| **pinned** | something that must remain true | **the code is wrong — fix the code** |

## The rule

**Never edit a pinned test to agree with new behaviour.** If a pinned test
fails, the change under your hands broke an invariant. Fix the change.

Relaxing or deleting a pinned assertion is a decision that the invariant no
longer holds. That is a design change, and it belongs to the user — say what
broke and why you think the invariant should move, and let them decide. Do not
make that call while clearing a test failure.

**Never touch a pinned test as part of a bulk edit.** No sweep, no regex, no
"update every call site" pass may include one. A mechanical sweep is exactly
how a real regression gets carried along with the legitimate updates: the
pinned test goes red for the true reason, the sweep rewrites it with all the
others, and the one signal that would have caught the bug is gone. If a sweep
would touch a pinned test, stop and handle that test by itself, deliberately,
and say what you concluded.

There is no enforcement, and there cannot be: a marker cannot stop an edit. It
makes the distinction visible in the failure output, at the moment someone is
deciding what to change.

## What to pin

Pin a property that would still have to be true after a redesign:

- **Identity and its consequences** — the same configuration reached two ways
  is one block; `hash` is `sha256(type())`; a signature is relocatable.
- **What must not silently change meaning** — a specline is what lands in the
  identity, not its resolved value; two slices carrying one column name both
  survive.
- **A guard that gives other tests teeth** — that distinct things stay
  distinct, so a parity test cannot pass by everything collapsing to one value.
- **A bug that cost real time**, expressed as the property it violated.

Do not pin a rendering, an argument name, a property-vs-method choice, an error
message, or a column order. Those are decisions, and decisions get revisited.

## Writing one

Say the invariant in the docstring, not the mechanics — the reader needs to
know what would be lost if the assertion were deleted.

```python
@pytest.mark.pinned
class TestSpeclinesCommute:
    """A spec value that is not its own value.

    `var.source` is the RESOLVED path and `spec['source']` the unexpanded
    text. A route that rebuilds from the resolved value produces a block that
    is configured identically and identified differently -- and every check
    that compares configuration would pass.
    """
```

Pin the guard alongside the property. A parity test that passes because both
sides collapsed to one value is worse than no test, so pin the distinctness
too.
