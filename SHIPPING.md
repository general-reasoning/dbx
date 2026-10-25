# Shipping

Copying topics **out** of dbx, to readers who have no dbx — a config checked into
another repo, an artifact in a shared directory, a table on Azure.

```python
datablock.UNSAFE_ship(configs="~/behavior_cloning/configs/data")        # copy out
datablock.UNSAFE_ship_redirect(facts="abfss://research@acct/bc/facts.json")  # read from there
```

## `UNSAFE_ship(targets=None, /, **topic_targets)`

The outbound mirror of `UNSAFE_copy_from`: that one takes a single source
anchorkeypath and copies every topic **in**; this takes a target **per topic** and
copies only the topics named. Returns `{topic: target}` as shipped.

Targets go positionally as a dict or as keywords, so a topic whose name collides with
an option still has a spelling. Either side may be local or an fsspec URL.

| | |
|---|---|
| a topic left out | not shipped — a datablock ships its config and keeps its checkpoints |
| a `SYNTHETIC` topic | ships nothing; it has no location |
| a topic group | ships its leaves under the target as a directory |
| an unknown topic, or no topic at all | `ValueError` |
| a target that exists | `FileExistsError` unless `overwrite=True` |
| an invalid datablock | `AssertionError` — shipping sends data out, and an unbuilt datablock has none |

That last one is the deliberate inverse of `UNSAFE_copy_from`, which asserts the
datablock is *not* valid before copying in.

The targets are journalled (`UNSAFE_ship:BEGIN` / `:END`). That is what makes a shipped
file traceable: a yaml checked into another repo is otherwise anonymous, and nothing
else says which datablock's VAR produced it.

## `UNSAFE_ship_redirect(...)`

Ships, then `UNSAFE_redirect`s the shipped topics at where they landed. After it the
shipped copy is the single source of truth — `read()` and `path()` answer through it,
so the datablock and its readers see one copy rather than two that drift. Editing the
shipped file changes what `read()` returns; deleting the datablock's own copy changes
nothing.

Partial, like every redirection: topics not named keep reading their own data. It
passes `merge=True`, so shipping a second topic later does not drop the first
redirection.

## Extending

`_UNSAFE_ship_topic_(topic, target, *, overwrite=False)` is the per-topic workhorse and
the extension point — override it to ship a subset of a topic's files, the way
`_UNSAFE_copy_topic_` is overridden to copy one. It dispatches dir-vs-file through
`_is_dir_topic_` and copies with `_UNSAFE_copy_fs_`, which handles local and remote on
either side and does a server-side blob copy when both are the same filesystem.

Tests: `tests/test_ship.py`.
