"""dbx.stills — :class:`Still`, the training-run Datablock.

A *still* is one training run, addressed by its configuration: its weights and
its TensorBoard logs live under a key derived from the hash of everything that
produced them.  Every such run in this codebase had grown its own copy of the
same ~700 lines -- the ``_COMPLETE`` marker, the checkpoint
save/upload/free/resume dance, the log symlink, the atexit sync, the
``UNSAFE_*`` helpers and the Lightning ``Trainer`` assembly.  This module holds
one of each.

Not imported by ``dbx/__init__.py``
-----------------------------------
``import dbx`` must work without torch, and this module needs both torch and
lightning at module scope.  Import it by name -- ``from dbx.stills import
Still`` -- exactly as :mod:`dbx.datastreams` is imported.

Two ways to say what to train
-----------------------------
**Explicit mode** puts a Datablock in ``VAR.lightning`` and reads
``.lightning_module`` off it -- one more block in the tree, with an identity of
its own, which is what you want when the module's own construction has upstream
artifacts (downloaded weights, a warm-start still).
:class:`LightningBuilder` is the base for that block.

**cfg mode** names two ordinary classes -- ``Model`` and ``Lightning``, which
import nothing from dbx -- and mirrors their ``cfg_``-prefixed keyword
arguments into this block's own ``VAR``.

``Still.Lightning`` is the cfg-mode attribute and :class:`LightningBuilder`
the explicit-mode block; the Builder suffix is what keeps the bare word free
for the attribute, which is the one a still author writes::

    class MyModel(nn.Module):
        def __init__(self, *, cfg_embed_dim=768, cfg_depth=12): ...

    class MyLightning(L.LightningModule):
        def __init__(self, model, *, cfg_learning_rate=1e-4): ...
        def train_transform(self): ...
        def train_collate_fn(self): ...

    class MyStill(Still):
        VERSION = 1
        Model, Lightning = MyModel, MyLightning

        @dataclass
        class VAR(Still.VAR):
            embed_dim: int = 768
            depth: int = 12
            learning_rate: float = 1e-4

``__build__`` then calls ``MyModel(cfg_embed_dim=var.embed_dim, ...)`` and
``MyLightning(model, cfg_learning_rate=var.learning_rate)``.  A name declared
on **both** classes is one VAR field passed to both, which is the right
reading: ``crop_size`` genuinely is one knob two objects have to agree about.

Write that VAR by hand or generate it with :func:`scaffold_still`; either way
it is source in your repo, which is deliberate -- see below.

Why the VAR is written down and not reflected
---------------------------------------------
``Datablock._typed_specdict`` walks **every** ``VAR.__dataclass_fields__``, at
its default, into the signature -- so into the hash, so into the storage key.
A VAR derived from ``Model.__init__`` at import time therefore means that
adding one ``cfg_`` knob, or changing one ``cfg_`` default, silently re-keys
every run of that model.  The author has no reason to expect it: their file
does not mention dbx.

So the VAR is checked-in source.  A hash movement then shows up in a diff,
where it can be argued with.  What is checked at runtime is only that the two
have not drifted apart: :meth:`Still.__check_cfg__` raises if a ``cfg_``
argument exists that no VAR field feeds, which turns a silent re-keying into a
loud construction error.

The identity-affecting half of ``VAR`` is the training configuration
(``max_epochs``, ``batch_size``, ``precision``, the ``val_*`` knobs, ...).
Everything operational -- device count, worker count, where TensorBoard logs
get symlinked -- is a constructor keyword and affects no hash.
"""

from __future__ import annotations

import atexit
import inspect
import os
import re
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import fsspec
import torch

import lightning as L
import lightning.pytorch.callbacks
import lightning.pytorch.loggers

from dbx.datablocks import Datablock
from dbx.dataparts import UNSAFE_allowed
from dbx.datastreams import (
    BlockShuffleSampler,
    ResumableDataLoader,
    block_split_indices,
    block_split_ranges,
    shuffled_block_order,
    val_loader_workers,
)

__all__ = [
    'Cfgparam',
    'LightningBuilder',
    'Still',
    'Weights',
    'cfg_params',
    'scaffold_still',
    # Re-exported from dbx.datastreams, which is torch-only: importing this
    # module pulls in lightning, and the split/sizing helpers are useful to
    # code that has no business doing that.
    'block_split_indices',
    'block_split_ranges',
    'shuffled_block_order',
    'val_loader_workers',
]


# ═══════════════════════════════════════════════════════════════════════
# The cfg_ protocol
# ═══════════════════════════════════════════════════════════════════════

def is_intact_archive(path) -> bool:
    """True if *path* is an existing, readable zip archive, as torch writes one.

    Existence alone is not enough: an interrupted download or a run killed
    mid-save leaves a file that passes ``exists()`` and then fails inside
    ``torch.load``, which for a resume means hours later.
    """
    if path is None or not os.path.isfile(path):
        return False
    try:
        with zipfile.ZipFile(path) as zf:
            return zf.testzip() is None
    except (zipfile.BadZipFile, EOFError, OSError):
        return False


CFG_PREFIX = 'cfg_'


@dataclass(frozen=True)
class Cfgparam:
    """One ``cfg_`` knob of a user-provided class, as the VAR field it becomes.

    ``name`` is the ``cfg_``-stripped name, which is what the VAR field is
    called.  ``roles`` names every constructor that takes it -- two entries
    when ``Model`` and ``Lightning`` both declare it, in which case one VAR
    field feeds both.
    """

    name: str
    default: Any
    annotation: Any
    roles: tuple

    @property
    def cfg_name(self) -> str:
        return f"{CFG_PREFIX}{self.name}"

    def annotation_text(self) -> str:
        """The annotation as source text, for a generated dataclass field."""
        a = self.annotation
        if a is inspect.Parameter.empty:
            return 'object'
        if isinstance(a, str):
            return a
        return getattr(a, '__name__', None) or str(a).replace('typing.', '')


def cfg_params(roles: dict, *, prefix: str = CFG_PREFIX) -> dict:
    """The ``cfg_`` surface of *roles*, as ``{stripped name: Cfgparam}``.

    *roles* maps a role name to a class (or to ``None``, which is skipped):
    ``{'model': MyModel, 'lightning': MyLightning}``.  Declaration order is
    preserved, role by role, so a generated VAR reads in the order the author
    wrote their arguments.

    A name declared in more than one role must agree on its default, since one
    VAR field supplies both; disagreement raises rather than picking a winner.
    """
    out: dict = {}
    for role, cls in roles.items():
        if cls is None:
            continue
        try:
            signature = inspect.signature(cls.__init__)
        except (TypeError, ValueError) as e:
            raise TypeError(f"cannot introspect {cls!r}.__init__: {e}") from e
        for pname, param in signature.parameters.items():
            if not pname.startswith(prefix):
                continue
            if param.kind not in (param.KEYWORD_ONLY, param.POSITIONAL_OR_KEYWORD):
                raise TypeError(
                    f"{cls.__name__}.__init__: {pname} is {param.kind.description}; "
                    f"a {prefix} argument must be passable by keyword"
                )
            name = pname[len(prefix):]
            if not name:
                raise TypeError(f"{cls.__name__}.__init__: bare {prefix!r} is not a name")
            existing = out.get(name)
            if existing is None:
                out[name] = Cfgparam(
                    name=name,
                    default=param.default,
                    annotation=param.annotation,
                    roles=(role,),
                )
                continue
            if existing.default != param.default:
                raise TypeError(
                    f"{pname} is declared by more than one of {list(roles)} with "
                    f"different defaults ({existing.default!r} vs {param.default!r}). "
                    f"One VAR field supplies both, so they must agree -- make the "
                    f"defaults match, or rename one of them"
                )
            out[name] = Cfgparam(
                name=existing.name,
                default=existing.default,
                annotation=(existing.annotation
                            if existing.annotation is not inspect.Parameter.empty
                            else param.annotation),
                roles=existing.roles + (role,),
            )
    return out


# ═══════════════════════════════════════════════════════════════════════
# LightningBuilder — the Datablock that owns a LightningModule
# ═══════════════════════════════════════════════════════════════════════

class LightningBuilder(Datablock):
    """A Datablock whose whole job is to build one ``LightningModule``.

    It writes nothing.  It exists so that a module's *construction* --
    architecture, optimiser schedule, loss configuration, and whichever
    upstream blocks supply its initial weights -- has an identity of its own,
    which the still that trains it then depends on.  Subclass it and implement
    :meth:`__lightning_module__`.

    Subclasses must not override :meth:`valid` or :meth:`__build__` without
    reading why they are what they are below.
    """

    VERSION = 1

    # No TOPICS, deliberately, and NOT `TOPICS = []`: the two render
    # differently into the identity -- signature_topics() answers
    # ("topics:None",) for a class with no TOPICS and () for one declaring an
    # empty list -- so declaring the empty list here would move the hash of
    # every subclass that already has artifacts on disk.

    # 1. Datablock protocol ─────────────────────────────────────────

    def valid(self):
        """True only when every nested Datablock in VAR has actually been built.

        A block with no topics is "always valid" under the base default, and
        ``build_tree()``'s shallow-skip optimisation would then skip this whole
        subtree unconditionally -- never building the upstreams in ``VAR`` that
        :meth:`__lightning_module__` needs, and surfacing much later as a
        ``FileNotFoundError`` from a missing weights file.

        Delegated generically through ``valid_var()``, which walks every VAR
        field that is a Datablock, rather than naming them: a VAR that grows
        another upstream stays correct without an edit here.
        """
        return self.validtopics(reduce=True) and self.valid_var(reduce=True)

    def __build__(self):
        """Unreachable through ``build()`` -- reaching it is itself the bug report.

        There is nothing to build: the upstreams in ``VAR`` are built by
        ``build_tree()``'s bottom-up recursion, and this block only reads them.
        Two things already stop ``build()`` short of here, and it raises rather
        than inheriting the base's silent no-op so that anything getting PAST
        both of them fails loudly instead of "succeeding" with the upstreams
        never built -- which would surface as a confusing ``FileNotFoundError``
        minutes into a run:

        * with ``validate_vars=True`` (the default) an unbuilt upstream makes
          :meth:`valid` False and ``__pre_build__`` raises;
        * with ``validate_vars=False``, ``valid_var()`` answers True
          unconditionally, so :meth:`valid` is True and ``build()`` skips the
          block as already done.

        So this fires only for a subclass that overrides :meth:`valid` into
        something falsifiable while leaving ``__build__`` alone, or for a
        direct ``__build__()`` call. Both are bugs, and both are worth naming.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.__build__() should be unreachable: it has no "
            f"topics and no build of its own, only components in VAR that must be "
            f"built by build_tree()'s bottom-up recursion. If you are seeing this, "
            f"something called .build()/__build__() where .build_tree() was needed."
        )

    # 2. Properties and accessors ───────────────────────────────────

    @property
    def lightning_module(self):
        """The ``LightningModule``, built once and kept.

        Cached on the instance rather than via ``functools.cached_property``
        so that a subclass can override :meth:`__lightning_module__` -- the
        thing that actually builds it -- without restating the caching.
        """
        if getattr(self, '_lightning_module', None) is None:
            self._lightning_module = self.__lightning_module__()
        return self._lightning_module

    def __lightning_module__(self):
        """Build and return the ``LightningModule``.  Implement in a subclass."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement __lightning_module__()"
        )


# ═══════════════════════════════════════════════════════════════════════
# Weights — a weight file as an addressed artifact
# ═══════════════════════════════════════════════════════════════════════

class Weights(Datablock):
    """One set of pre-trained weights, persisted under its own identity.

    A model class should not fetch its own weights: where they came from is an
    upstream artifact with a provenance worth addressing, and a class that
    downloads in its constructor cannot be handed to anyone who lacks the
    credentials.  So this block resolves ``var.ckpt`` -- a URL, a local path,
    or ``None`` for random initialisation -- to a persisted file once, and
    :meth:`path_local` hands the consumer a **local path**, which is the
    only thing ``torch.load`` will take.

    Subclass it to teach it a naming scheme (a registry of published
    checkpoints, say) by overriding :meth:`source_url`.
    """

    VERSION = 1

    # The dict form, naming the file, NOT `TOPICS = ['weights']`: a list entry
    # is a DIRECTORY topic, so path('weights') answers the directory itself and
    # `ensure_dirpath=True` creates it -- after which opening that same path
    # for writing is IsADirectoryError on any real filesystem. (It survives on
    # blob storage only because there are no directories there to collide
    # with.) Naming the file makes path() a file and dirpath() its parent.
    TOPICS = {'weights': 'weights.pt'}

    @dataclass
    class VAR(Datablock.VAR):
        ckpt: Optional[str] = None   # registry key, url, local path, or None

    # 1. Datablock protocol ─────────────────────────────────────────

    def valid(self):
        """True when the weights are reachable, or none are wanted.

        ``ckpt=None`` means random initialisation: there is nothing to fetch,
        so the block is trivially satisfied.

        Otherwise the *remote blob* must exist -- checked directly rather than
        trusting the journal, because a build whose upload failed leaves the
        block recorded as built with nothing behind it, and ``build_tree()``
        would then skip the download and fail at load time instead.  A local
        copy that is intact also counts, and one that is not is deleted here
        so the next :meth:`path_local` refetches rather than loading a
        truncated file.
        """
        if self.var.ckpt is None:
            return True
        local = self.path('weights', local=True)
        if self._is_intact(local):
            return True
        if local is not None and os.path.isfile(local):
            self.log.warning("valid(): corrupt local weights, removing: %s", local)
            os.remove(local)
        remote = self.path('weights')
        if remote is None:
            return False
        try:
            return self.fs.exists(remote)
        except Exception:
            return False

    def __build__(self):
        """Fetch the weights into the ``weights`` topic, unless none are wanted."""
        if self.var.ckpt is None:
            self.log.info("Weights: ckpt=None -- random initialisation, nothing to fetch")
            return self

        url = self.source_url()
        dest = self.path('weights', ensure_dirpath=True)
        self.log.info("Weights: fetching %s -> %s", url, dest)
        # Streamed in chunks rather than read whole: these are multi-GB files,
        # and fsspec.open takes the http(s)/abfs(s)/file url a registry might
        # name, which self.pull() cannot -- pull's source must already be a
        # path on this block's own filesystem.
        with fsspec.open(url, 'rb') as src, self.fs.open(dest, 'wb') as dst:
            while True:
                chunk = src.read(self.CHUNK_BYTES)
                if not chunk:
                    break
                dst.write(chunk)
        self.log.info("Weights: persisted %s", dest)
        return self

    # 2. Properties and accessors ──────────────────────────────

    def source_url(self) -> str:
        """Where ``var.ckpt`` is to be fetched from.  Override to add a registry.

        The default reads it literally, so ``ckpt`` may be any url ``fsspec``
        understands (including a plain local path).  A subclass that publishes
        named checkpoints maps the name here, and raises on an unknown one.
        """
        return str(self.var.ckpt)

    def path_local(self) -> Optional[str]:
        """A local path to the weights, downloading them if need be, or ``None``.

        ``None`` exactly when ``var.ckpt`` is ``None`` -- when the model is to
        be randomly initialised -- so a caller can pass the result straight
        through as a ``cfg_ckpt`` argument either way, with no branch of its
        own.

        ``torch.load`` needs a real file, which is the whole reason this is
        separate from :meth:`Datablock.path`: that one names the canonical
        (possibly remote) location.
        """
        if self.var.ckpt is None:
            return None
        dest = self.path('weights', local=True)
        if not self._is_intact(dest):
            self.pull(self.path('weights'), dest, show_progress=True)
        if not self._is_intact(dest):
            raise RuntimeError(
                f"Weights: {dest} is not a readable archive after download "
                f"from {self.path('weights')}; the persisted blob is truncated -- "
                f"UNSAFE_clear() this block and rebuild it"
            )
        return dest

    # 3. Private and utility methods ───────────────────────────

    #: Copy buffer for the streamed fetch. 64 MiB: large enough that a
    #: multi-GB transfer is not dominated by per-chunk overhead.
    CHUNK_BYTES = 64 * 1024 * 1024

    _is_intact = staticmethod(is_intact_archive)


# ═══════════════════════════════════════════════════════════════════════
# Still — one training run
# ═══════════════════════════════════════════════════════════════════════

class Still(Datablock):
    """One Lightning training run, addressed by its configuration.

    Two topics: ``ckpts`` (the checkpoints, plus a ``_COMPLETE`` marker) and
    ``logs`` (the TensorBoard run directories).  Both are staged locally and
    synced to the block's storage as the run proceeds, so a run whose machine
    dies leaves its checkpoints behind, and a resumed run finds them.

    What a subclass has to supply
    -----------------------------
    * The module to train -- either ``VAR.lightning`` holding a
      :class:`LightningBuilder`, or ``Model``/``Lightning`` class attributes and
      the matching ``cfg_`` VAR fields (see the module docstring).
    * A dataset builder in ``VAR.training_dataset_builder``: anything with a
      ``.dataset(transform=...)`` method.  ``VAR.validation_dataset_builder``
      is optional -- without it, validation is a held-out block-granular split
      of the training builder's own dataset.
    * ``train_transform`` / ``val_transform`` / ``train_collate_fn`` /
      ``val_collate_fn`` on the LightningModule, if it wants any.  These are
      the model's business -- an augmentation pipeline and a collator are as
      model-specific as the forward pass -- so they live with the model rather
      than being configured here.  Each defaults to ``None``, i.e. plain
      tensors and the default collate.

    What it may override
    --------------------
    :meth:`_banner_rows` (what the start-of-run banner reports),
    :meth:`dataloaders`, :meth:`trainer_kwargs`, :meth:`callbacks`, and the
    ``Sampler`` / ``Loader`` / ``split_indices`` hooks.

    Operational parameters -- ``n_devices``, ``num_workers``,
    ``prefetch_factor``, ``tensorlogs_root``, ``save_remote_logs``,
    ``stall_timeout_s``, ``debug_share_train_val`` -- are constructor keywords,
    not VAR: they change how a run executes, not what it computes, so two runs
    differing only in those are the same run and share a key.
    """

    VERSION = 1
    TOPICS = ['ckpts', 'logs']

    # A warm-start source in `ckpt` needs *a* checkpoint, not a finished run:
    # branching off a still that is still training, or that stopped before
    # writing its _COMPLETE marker, is a legitimate and common thing to do.
    # BUILD_TREE_EXEMPTIONS keeps build_tree() from training the source on your
    # behalf; TREE_SKIP_VALIDATION keeps __pre_build__'s validate_vars pass from
    # rejecting the whole build with a generic "not all upstream Datablocks in
    # var are valid" -- where __build__ would otherwise say precisely which
    # still has no checkpoints.
    BUILD_TREE_EXEMPTIONS = ('ckpt',)
    TREE_SKIP_VALIDATION = {'ckpt'}

    #: The classes cfg mode dispatches to.  Left None in explicit mode, where
    #: ``VAR.lightning`` supplies the module instead.
    #:
    #: ``Lightning`` here is a plain ``LightningModule`` subclass, not the
    #: :class:`LightningBuilder` block that wraps one -- which is what
    #: ``VAR.lightning`` holds, and why that class carries the suffix.
    Model: Optional[type] = None
    Lightning: Optional[type] = None

    #: Sampler and loader for :meth:`dataloaders`.  Both default to the
    #: shard-locality-aware pair in :mod:`dbx.datastreams`; a subclass whose
    #: data is not shard-backed can drop to ``torch.utils.data`` equivalents.
    Sampler: type = BlockShuffleSampler
    Loader: type = ResumableDataLoader

    #: ``(n, block_size, fractions, seed=) -> [indices, ...]``, used to draw the
    #: validation split when there is no separate validation builder.
    split_indices = staticmethod(block_split_indices)

    BANNER_TITLE = " TRAINING START "
    #: Double-width glyphs in BANNER_TITLE, to compensate its centering: an
    #: emoji occupies two terminal columns but counts as one character, so
    #: str.center() lands one column off per emoji.
    BANNER_TITLE_EMOJI = 0
    #: Inner width, between the borders.  Every row is truncated to fit, so
    #: this is the budget for the long values -- resolved checkpoint paths, the
    #: dataset tag, the block key -- which at 76 were all cut to "...".
    BANNER_WIDTH = 152

    @dataclass
    class VAR(Datablock.VAR):
        # FIELD ORDER IS LOAD-BEARING for a subclass carrying LEGACY_NORM:
        # __expand_spec__ renders a legacy block's spec in __dataclass_fields__
        # order (and sorts only in the modern rendering), so reordering these
        # re-keys every such block. A new field appended by a subclass is safe;
        # inserting one here is not.
        #
        # --- What to train ---
        lightning: object = None                    # a LightningBuilder, in explicit mode
        training_dataset_builder: object = None     # anything with .dataset(transform=)
        validation_dataset_builder: object = None   # optional; else a split of the above
        train_val_split: float = 0.8
        dataset_seed: int = 42
        block_shuffle_size: int = 2048   # samples per shuffle block; see Sampler
        # --- Where to start ---
        # None: resume this run's own latest checkpoint, if it has one.
        # A block: warm-start from *its* latest -- weights only, at step 0,
        #   fresh optimizer, since another run's step count and moments say
        #   nothing about this one. Consulted only until this run has a
        #   checkpoint of its own; after that a resume continues this run.
        # A path: resume it in full, optimizer and step count included, which
        #   is what you want when pointing a run at its own checkpoint that
        #   has moved.
        # See _resolve_resume_ckpt().
        ckpt: object = None
        # Misnomer kept for identity stability: it means "ignore any resume
        # checkpoint of MY OWN and restart the run at step 0", NOT "randomly
        # initialise the weights". Weight init is the model's business.
        from_scratch: bool = False
        # Force the weights-only load for *any* checkpoint, this run's own
        # included -- so a run that must stay resumable should not leave it on
        # merely to get a clean start on its first launch. A warm start from a
        # block already loads that way without it.
        reset_optimizer_state: bool = False
        # --- How long ---
        max_epochs: int = 30
        max_training_steps: Optional[int] = None
        # Every *_n_steps field below counts OPTIMIZER steps, not micro-batches
        # -- see the val_check_interval scaling in trainer_kwargs(). 1 = every step.
        train_log_every_n_steps: int = 10
        val_every_n_steps: int = 500
        val_max_batches: Optional[int] = 1
        val_shuffle: bool = False
        val_seed: int = 42
        gradient_clip_val: float = 1.0
        gradient_clip_algorithm: str = 'norm'
        ckpt_every_n_steps: Optional[int] = None
        ckpt_every_n_epochs: int = 5
        precision: Optional[str] = 'bf16-mixed'
        matmul_precision: Optional[str] = 'high'
        accumulate_grad_batches: int = 2
        batch_size: int = 16
        still_seed: int = 42

    #: VAR names Still itself defines. A cfg_ argument may not take one:
    #: the meanings differ (`cfg_ckpt` on a model is its initial weights;
    #: `VAR.ckpt` on a still is the run it resumes from), and one field cannot
    #: carry both. Computed rather than listed, so it cannot fall behind VAR.
    RESERVED_VAR_NAMES = frozenset(VAR.__dataclass_fields__)

    # 1. Datablock protocol ─────────────────────────────────────────

    def __init__(
        self,
        *,
        n_devices=1,
        tensorlogs_root=None,
        # Accepted and currently wired to nothing. Every still in this
        # codebase has taken it since before the weights-only resume path
        # existed, and that path deliberately loads with strict=False. Kept as
        # an explicit parameter rather than dropped: it is recorded in the
        # blocks' journalled dfn, so removing it would stop a recorded block
        # from reconstructing.
        strict_loading=False,
        save_remote_logs=True,
        debug_share_train_val=False,
        num_workers=4,
        prefetch_factor=4,   # PyTorch's default is 2; bumped to smooth I/O-bound stalls
        # Seconds one dataset item may take before its worker is declared
        # wedged. 0 or None disables it. Only does anything if the subclass
        # returns a watchdog from worker_init_fn().
        stall_timeout_s=None,
        **kwargs,
    ):
        # Threaded through super() rather than assigned directly: the base
        # records **kwargs in self.parameters, which is what __getstate__
        # serializes, so .set(tag=...) preserves them. Plain `self.x = ...`
        # would silently vanish on rebuild -- and every pipeline entrypoint
        # calls .set(tag=...).
        super().__init__(
            n_devices=n_devices,
            tensorlogs_root=tensorlogs_root,
            strict_loading=strict_loading,
            save_remote_logs=save_remote_logs,
            debug_share_train_val=debug_share_train_val,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            stall_timeout_s=stall_timeout_s,
            **kwargs,
        )

    def __post_init__(self):
        super().__post_init__()
        if self.Model is not None or self.Lightning is not None:
            self.__check_cfg__()

    def valid(self):
        """True only when training has run to completion.

        The ``_COMPLETE`` marker, not "are there checkpoints": a run that
        stopped at epoch 3 of 30 has checkpoints and is not done, and treating
        it as valid would make ``build_tree()`` hand a half-trained encoder to
        everything downstream.  Local marker first, then remote, so a run that
        finished on this machine answers without a network round trip.
        """
        local_marker = os.path.join(self._local_ckpts_dir, "_COMPLETE")
        if os.path.exists(local_marker):
            return True
        try:
            remote_ckpts = self.dirpath("ckpts")
            return self.fs.exists(os.path.join(remote_ckpts, "_COMPLETE"))
        except Exception:
            return False

    # 2. The module under training ──────────────────────────────────

    @property
    def lightning_module(self):
        """The ``LightningModule`` to train, built once and kept.

        Resolves whichever of the two modes this still is in: ``VAR.lightning``
        when it holds a block, else ``Model``/``Lightning`` with the ``cfg_``
        arguments projected out of VAR.
        """
        if getattr(self, '_lightning_module', None) is None:
            if self.var.lightning is not None:
                self._lightning_module = self.var.lightning.lightning_module
            else:
                self._lightning_module = self.__lightning_module__()
        return self._lightning_module

    def __lightning_module__(self):
        """Build the module from ``Model``/``Lightning`` and the cfg VAR fields."""
        if self.Lightning is None:
            raise ValueError(
                f"{type(self).__name__} has neither VAR.lightning nor a Lightning "
                f"class: nothing to train. Put a LightningBuilder in VAR.lightning, or "
                f"set the Model/Lightning class attributes (see dbx.stills)"
            )
        model = self.model()
        return self.Lightning(model, **self.lightning_cfg)

    def model(self):
        """Instantiate ``Model`` with the cfg VAR fields it declares."""
        if self.Model is None:
            raise ValueError(f"{type(self).__name__} has no Model class")
        return self.Model(**self.model_cfg)

    @classmethod
    def cfg_params(cls) -> dict:
        """The ``cfg_`` surface of ``Model`` and ``Lightning``, by stripped name."""
        return cfg_params({'model': cls.Model, 'lightning': cls.Lightning})

    def _cfg_for(self, role: str) -> dict:
        return {
            p.cfg_name: getattr(self.var, p.name)
            for p in self.cfg_params().values()
            if role in p.roles
        }

    @property
    def model_cfg(self) -> dict:
        """``{cfg_name: value}`` for every ``cfg_`` argument ``Model`` declares."""
        return self._cfg_for('model')

    @property
    def lightning_cfg(self) -> dict:
        """``{cfg_name: value}`` for every ``cfg_`` argument ``Lightning`` declares."""
        return self._cfg_for('lightning')

    def __check_cfg__(self):
        """Refuse to construct if VAR and the ``cfg_`` surface have drifted apart.

        The VAR is written down (see the module docstring) precisely so that a
        hash movement is visible in a diff.  The cost of writing it down is
        that it can fall behind the classes it mirrors -- a ``cfg_`` argument
        added upstream would otherwise be silently left at *its own* default,
        with the still's identity claiming to describe a configuration it does
        not actually pass.  So the two are compared here, on every
        construction, and a gap is a construction error naming the fields to
        add.  Which is also the hash-movement notice: adding them re-keys the
        block, and that is now a decision rather than an accident.
        """
        params = self.cfg_params()
        reserved = sorted(n for n in params if n in self.RESERVED_VAR_NAMES)
        if reserved:
            raise TypeError(
                f"{type(self).__name__}: {[CFG_PREFIX + n for n in reserved]} would "
                f"land on VAR field(s) {reserved}, which {Still.__name__} already "
                f"defines and means something else -- `ckpt` is the checkpoint this "
                f"RUN resumes from, `batch_size` is the loader's. One field cannot be "
                f"both. Rename the {CFG_PREFIX} argument(s) on your Model/Lightning "
                f"({CFG_PREFIX}init_ckpt, say)"
            )
        declared = set(type(self).VAR.__dataclass_fields__)
        missing = [p.name for p in params.values() if p.name not in declared]
        if missing:
            raise TypeError(
                f"{type(self).__name__}.VAR is missing a field for each of these "
                f"{CFG_PREFIX} arguments: {missing}. They would be left at their own "
                f"defaults and go unrecorded in this block's identity. Add them to "
                f"VAR (note that doing so re-keys this block, since every VAR field "
                f"is hashed), or regenerate the class with "
                f"dbx.stills.scaffold_still()"
            )

    # 3. Local working directories ──────────────────────────────────
    #
    # Backed by dbx's local= staging (Datablock.localanchorkeypath /
    # .dirpath(topic, local=True)) rather than hand-parsing anchorkeypath.
    # When the primary url is itself local storage, dbx aliases localfs/localroot
    # to fs/root, so these transparently collapse to the non-local
    # .path()/.dirpath() equivalents -- no branching needed here for that case.

    @property
    def _local_workdir(self):
        return self.localanchorkeypath

    @property
    def _local_logs_dir(self):
        return self.dirpath('logs', local=True)

    @property
    def _local_ckpts_dir(self):
        return self.dirpath('ckpts', local=True)

    @property
    def _logslink(self):
        if self.tensorlogs_root is None:
            return None
        return os.path.join(self.tensorlogs_root, self.key)

    def linklogs(self):
        """Symlink this run's local log dir under ``tensorlogs_root``.

        So that one ``tensorboard --logdir`` sees every run, named by key,
        without the logs themselves moving out of the block's staging area.
        """
        return self.linklocal('logs', self._logslink)

    # 4. UNSAFE_ helpers ────────────────────────────────────────────

    def UNSAFE_complete(self, *, OVERRIDE: bool = False):
        """Write the ``_COMPLETE`` marker locally and remotely, forcing :meth:`valid`.

        :meth:`__build__` calls this once ``trainer.fit()`` returns.  It is
        also directly callable to hand-declare a run complete -- e.g. after
        assembling checkpoints with :meth:`UNSAFE_copy_from` from a source
        whose ``ckpts`` carried no marker of its own.

        Each location is written directly rather than by syncing the whole
        ``ckpts`` topic, because this has to be safe to call when local
        staging holds no checkpoints at all (right after a copy that went
        straight to remote), where a local-to-remote directory push would
        overwrite real remote checkpoints with an empty local directory.
        :meth:`valid` checks local then remote, so either alone would do;
        writing both keeps staging and storage consistent with each other,
        matching what a real run leaves behind.
        """
        if not UNSAFE_allowed("UNSAFE_complete", OVERRIDE=OVERRIDE):
            return self
        timestamp = datetime.now().isoformat() + "\n"
        os.makedirs(self._local_ckpts_dir, exist_ok=True)
        with open(os.path.join(self._local_ckpts_dir, "_COMPLETE"), "w") as f:
            f.write(timestamp)
        if not self.is_local_fs:
            remote_marker = os.path.join(self.dirpath('ckpts', ensure=True), "_COMPLETE")
            with self.fs.open(remote_marker, "w") as f:
                f.write(timestamp)
        self.log.info(
            "UNSAFE_complete: wrote _COMPLETE marker (%s)",
            "local" if self.is_local_fs else "local + remote",
        )
        return self

    def UNSAFE_clear(self, *topics, OVERRIDE: bool = False, clear_dirpath: bool = False):
        """As the base, plus the local staging dirs and the TensorBoard symlink.

        Without this, clearing ``ckpts`` removes the remote checkpoints and
        leaves the local ones -- so :meth:`valid` still finds the local
        ``_COMPLETE`` marker and the block reports itself built.
        """
        result = super().UNSAFE_clear(*topics, OVERRIDE=OVERRIDE, clear_dirpath=clear_dirpath)
        if len(topics) == 0 or 'logs' in topics:
            local_logs = self._local_logs_dir
            if os.path.isdir(local_logs):
                shutil.rmtree(local_logs, ignore_errors=True)
            logslink = self._logslink
            if logslink and os.path.lexists(logslink):
                try:
                    os.remove(logslink)
                except OSError:
                    pass
        if len(topics) == 0 or 'ckpts' in topics:
            local_ckpts = self._local_ckpts_dir
            if os.path.isdir(local_ckpts):
                shutil.rmtree(local_ckpts, ignore_errors=True)
        return result

    def UNSAFE_clear_cache(self, *, OVERRIDE: bool = False):
        """Remove the local staging caches without touching remote storage.

        Deletes :attr:`_local_workdir` (the ``ckpts/`` and ``logs/`` subdirs
        under dbx's ``local=`` staging root) and the TensorBoard symlink.  The
        remote topics are left intact, so the run's record survives -- call
        this on a machine that has finished training and needs its disk back.
        """
        if not UNSAFE_allowed("UNSAFE_clear_cache", OVERRIDE=OVERRIDE):
            return self
        workdir = self._local_workdir
        if os.path.isdir(workdir):
            self.log.info("UNSAFE_clear_cache: removing local workdir %s", workdir)
            shutil.rmtree(workdir, ignore_errors=True)
        logslink = self._logslink
        if logslink and os.path.lexists(logslink):
            self.log.info("UNSAFE_clear_cache: removing logslink %s", logslink)
            try:
                os.remove(logslink)
            except OSError as e:
                self.log.warning("UNSAFE_clear_cache: could not remove logslink: %s", e)
        return self

    def UNSAFE_copy_from(self, anchorkeypath, *, ckpts: int = 0, **kwargs):
        """As :meth:`~dbx.Datablock.UNSAFE_copy_from`, but ``ckpts`` may be a subset.

        Parameters
        ----------
        ckpts : int, default 0
            0 copies every file under the source ``ckpts`` topic, as the base
            does.  A positive N copies only the earliest N by training step; a
            negative N only the most recent N.  The ``_COMPLETE`` marker, when
            present, is always copied alongside whatever was selected, so
            post-copy :meth:`valid` (and the default ``validate=True``) still
            pass.
        **kwargs
            Forwarded to the base (``OVERRIDE``, ``overwrite``, ``topicpaths``,
            ``validate``, ``always_copy_whole_dirpath``, ``show_progress``).
        """
        return super().UNSAFE_copy_from(anchorkeypath, ckpts=ckpts, **kwargs)

    def _UNSAFE_copy_topic(self, topic, anchorkeypath, *, ckpts: int = 0, **kwargs):
        if topic != 'ckpts' or not ckpts or kwargs.get('always_copy_whole_dirpath'):
            return super()._UNSAFE_copy_topic(topic, anchorkeypath, **kwargs)
        self._UNSAFE_copy_ckpts_subset(anchorkeypath, count=ckpts,
                                       topicpaths=kwargs.get('topicpaths'))

    def _UNSAFE_copy_ckpts_subset(self, anchorkeypath, *, count, topicpaths=None):
        """Copy the earliest (``count > 0``) or latest (``count < 0``) checkpoints."""
        _src_path = topicpaths['ckpts'] if topicpaths is not None else 'ckpts'
        src_path = os.path.join(anchorkeypath, _src_path)
        src_fs, _ = self._url_to_fs(src_path)
        if not src_fs.exists(src_path):
            return
        entries = [os.path.basename(p.rstrip('/')) for p in src_fs.ls(src_path, detail=False)]
        ckpt_names = sorted((n for n in entries if n.endswith('.ckpt')), key=self._ckpt_step)
        selected = ckpt_names[:count] if count > 0 else ckpt_names[count:]
        self.log.verbose(
            f"UNSAFE_copy_from: ckpts={count} -> copying {len(selected)}/{len(ckpt_names)} "
            f"checkpoint(s): {selected}"
        )
        dst_dir = self.dirpath('ckpts', ensure=True)
        for name in selected + (['_COMPLETE'] if '_COMPLETE' in entries else []):
            # _UNSAFE_copy_file prefers a direct server-side blob copy when src
            # and dst are on the same remote filesystem (the common case here:
            # two stills' ckpts/ under one storage account) -- which matters
            # because checkpoints are large, and the generic get-then-put
            # fallback round-trips every byte through local disk.
            self._UNSAFE_copy_file(
                os.path.join(src_path, name),
                os.path.join(dst_dir, name),
            )

    # 5. Checkpoints ────────────────────────────────────────────────

    #: Matches both ``step=1234`` and the doubled ``step=step=1234`` that
    #: Lightning's own ModelCheckpoint used to emit.
    CKPT_STEP_RE = re.compile(r'step=(?:step=)?(\d+)')

    @classmethod
    def _ckpt_step(cls, name):
        """The training step a checkpoint filename encodes, or -1."""
        m = cls.CKPT_STEP_RE.search(name)
        return int(m.group(1)) if m else -1

    @staticmethod
    def _is_intact_ckpt(path, log=None):
        """:func:`is_intact_archive`, saying which checkpoint it rejected and why."""
        ok = is_intact_archive(path)
        if not ok and log is not None:
            log.verbose(f"Skipping corrupt checkpoint {os.path.basename(str(path))}")
        return ok

    def find_latest_ckpt(self, *, pull: bool = False):
        """The most recent checkpoint, or ``None``.

        Parameters
        ----------
        pull : bool, default False
            When False, list the remote ``ckpts`` topic and return the latest
            entry's *remote* path by filename-derived step -- no download, no
            integrity check.  Cheap enough for an ad hoc "what is the latest
            checkpoint here?" from a one-liner, without pulling a multi-GB
            file just to look.

            When True, additionally sync that checkpoint to local storage
            (downloading if not already cached) and validate it as an intact
            archive, falling back to the next-latest if it is corrupt.  This
            is what an actual resume needs -- Lightning's ``ckpt_path=`` and
            ``torch.load`` both want a local file.
        """
        if not pull:
            return self._find_latest_ckpt_remote()

        os.makedirs(self.dirpath('ckpts', local=True), exist_ok=True)

        def _valid_ckpt(path):
            return self._is_intact_ckpt(path, log=self.log)

        try:
            result = self.synclocal(
                'ckpts', suffix='.ckpt', key=self._ckpt_step, validate=_valid_ckpt,
                latest=True, show_progress=True,
            )
        except Exception as e:
            self.log.verbose(
                f"find_latest_ckpt: remote check failed, falling back to local scan: {e}")
            result = None
            ckpts_dir = self.dirpath('ckpts', local=True)
            candidates = sorted(
                (f for f in os.listdir(ckpts_dir) if f.endswith(".ckpt")),
                key=self._ckpt_step,
            )
            for name in reversed(candidates):
                path = os.path.join(ckpts_dir, name)
                if _valid_ckpt(path):
                    result = path
                    break

        if result:
            self.log.info(
                "find_latest_ckpt: using %s (step=%s)",
                os.path.basename(result), self._ckpt_step(os.path.basename(result)),
            )
        return result

    def _find_latest_ckpt_remote(self):
        """The remote path of the latest ``.ckpt``, without pulling or validating."""
        try:
            entries = [(os.path.basename(e.rstrip('/')), e) for e in self.ls('ckpts')]
        except Exception as e:
            self.log.verbose(f"find_latest_ckpt(pull=False): remote listing failed: {e}")
            return None
        candidates = sorted(
            ((name, path) for name, path in entries if name.endswith('.ckpt')),
            key=lambda item: self._ckpt_step(item[0]),
        )
        if not candidates:
            return None
        latest_name, latest_path = candidates[-1]
        self.log.info(
            "find_latest_ckpt(pull=False): latest is %s (step=%s) at %s -- not downloaded",
            latest_name, self._ckpt_step(latest_name), latest_path,
        )
        return latest_path

    def _free_local_ckpt(self, local_path):
        """Delete a local checkpoint that is known to be duplicated on remote.

        Used for the checkpoint a run resumes *from*: :meth:`find_latest_ckpt`
        downloads it and nothing else ever cleans it up (unlike newly-saved
        checkpoints, which are freed right after their own upload), so it would
        otherwise hold disk for the whole run.  A no-op when storage is local
        (there would be no remote copy to fall back on) or the path is gone.
        """
        if self.is_local_fs:
            return
        try:
            if os.path.exists(local_path):
                os.remove(local_path)
                self.log.info("Freed local resume checkpoint: %s", local_path)
        except OSError as e:
            self.log.warning("Could not free local resume checkpoint: %s", e)

    # 6. Sync helpers ───────────────────────────────────────────────
    #
    # pushtopic(topic) pushes the whole local-staged topic dir to its canonical
    # remote path, no-oping on its own when they are the same path (local
    # storage) -- dbx's push() checks src == dest directly. The is_local_fs
    # guards below only skip the (otherwise misleading) "syncing to remote" log
    # line when there is no separate remote to sync to.

    def _sync_ckpts_to_remote(self):
        if self.is_local_fs:
            return
        self.log.info("Syncing ckpts to remote")
        self.pushtopic('ckpts')

    def _sync_logs_to_remote(self):
        if self.is_local_fs:
            return
        self.log.info("Syncing logs to remote")
        self.pushtopic('logs')

    def _sync_to_remote(self, *, reason):
        """Push both topics, never raising -- called from ``atexit`` and ``finally``."""
        try:
            self._sync_ckpts_to_remote()
        except Exception as e:
            self.log.warning("%s: ckpt sync failed: %s", reason, e)
        if self.save_remote_logs:
            try:
                self._sync_logs_to_remote()
            except Exception as e:
                self.log.warning("%s: log sync failed: %s", reason, e)

    # 7. Dataloaders ────────────────────────────────────────────────
    #
    # The transform and the collator are the MODEL's business -- an
    # augmentation pipeline and a collator are as model-specific as the forward
    # pass -- so they are asked of the LightningModule rather than configured
    # here. Every hook is optional: a module that implements none gets plain
    # items and torch's default collate.
    #
    # The one non-obvious hook is `shared_train_collate_fn`. When there is no
    # separate validation builder, train and val are two index subsets of ONE
    # dataset, so training items arrive carrying whatever extra payload
    # validation asked for (`val_dataset_kwargs`), and the training collator has
    # to drop it. With a separate validation builder the two datasets are built
    # independently and `train_collate_fn` applies as-is.

    #: Every hook :meth:`dataloaders` asks the module for. All optional. The
    #: canonical list, so that :meth:`module_hooks` can report what a module
    #: actually supplies -- worth having because a MISSPELLED hook is silently
    #: ignored, and the symptom is untransformed data rather than an error.
    MODULE_HOOKS = (
        'train_transform',
        'val_transform',
        'train_collate_fn',
        'val_collate_fn',
        'shared_train_collate_fn',
        'train_dataset_kwargs',
        'val_dataset_kwargs',
    )

    @classmethod
    def module_hooks(cls, module) -> dict:
        """``{hook: whether *module* supplies it}``, for every hook in use.

        Diagnostic: print it when a loader is yielding something unexpected.
        A hook whose name is a near-miss shows up here as the real one missing.
        """
        return {name: getattr(module, name, None) is not None
                for name in cls.MODULE_HOOKS}

    @staticmethod
    def _module_hook(module, name, default=None):
        """Call *module*'s *name* hook if it has one, else return *default*."""
        hook = getattr(module, name, None)
        if hook is None:
            return default
        return hook() if callable(hook) else hook

    def worker_init_fn(self):
        """The ``worker_init_fn`` every loader this still builds is given.

        ``None`` here, so plain torch behaviour.  Override to install a
        per-worker guard -- detaching inherited stdio, arming a stall watchdog.
        Whatever it returns must be picklable, since a ``spawn``-context
        DataLoader has to send it to the worker.
        """
        return None

    def _dataloader(self, dataset, *, batch_size, num_workers, prefetch_factor,
                    collate_fn, sampler, loader_cls=None):
        """Build one DataLoader.  The single place a loader is constructed.

        Both loaders come through here so that ``worker_init_fn`` cannot be
        passed to one and forgotten on the other -- which is exactly what
        happened before, and had no symptom until a forked worker deadlocked on
        a logging lock copied while held, possibly hours into a run.

        ``prefetch_factor`` is only a legal DataLoader argument when
        ``num_workers > 0``, hence the conditional rather than a default.
        """
        kwargs = {} if num_workers <= 0 else {'prefetch_factor': prefetch_factor}
        cls = loader_cls or torch.utils.data.DataLoader
        return cls(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
            sampler=sampler,
            worker_init_fn=self.worker_init_fn(),
            **kwargs,
        )

    def dataloaders(self, model=None, debug_share_train_val=None):
        """The ``(train, val)`` DataLoaders, exactly as :meth:`__build__` uses them.

        Runs no training -- useful for inspecting the dataset/collation wiring
        directly, e.g. to confirm the images reaching the model are meaningful.

        Parameters
        ----------
        model : LightningModule, optional
            Reused when already built (as during ``__build__``); otherwise
            taken from :attr:`lightning_module`, which caches, so repeated
            calls do not rebuild it.
        debug_share_train_val : bool, optional
            Diagnostic only -- never for a real run, it defeats held-out
            validation entirely.  When True (and no separate validation
            builder is configured), train and val both iterate the *same*
            dataset with no disjoint index split.  Useful for isolating
            whether slow loading comes from the train/val partition doubling
            the distinct shard working set -- two disjoint index pools that
            never overlap, so the cache must hold both -- rather than from
            something else.  Deliberately not a VAR field: it must never
            affect identity, since it is not a real training configuration.
            ``None`` (default) inherits the constructor's
            ``debug_share_train_val``.
        """
        if model is None:
            model = self.lightning_module
        if debug_share_train_val is None:
            debug_share_train_val = self.debug_share_train_val

        var = self.var
        train_builder = var.training_dataset_builder
        val_builder = var.validation_dataset_builder
        if train_builder is None:
            raise ValueError(
                f"{type(self).__name__}: VAR.training_dataset_builder is None; "
                f"there is nothing to train on"
            )

        self.log.debug("module hooks: %s", self.module_hooks(model))
        train_transform = self._module_hook(model, 'train_transform')
        val_transform = self._module_hook(model, 'val_transform')
        train_kwargs = self._module_hook(model, 'train_dataset_kwargs', {}) or {}
        val_kwargs = self._module_hook(model, 'val_dataset_kwargs', {}) or {}
        block_size = var.block_shuffle_size

        if val_builder is not None:
            train_dataset = train_builder.dataset(transform=train_transform, **train_kwargs)
            val_dataset = val_builder.dataset(transform=val_transform, **val_kwargs)
            train_collate_fn = self._module_hook(model, 'train_collate_fn')
        else:
            # ONE underlying dataset split into disjoint index subsets, rather
            # than two dataset instances over the same table: besides being the
            # correct way to draw a genuine holdout on the fly from a single
            # source, two live StreamingDatasets pointed at one table collide
            # on its local shard cache (mosaicml-streaming disallows two
            # instances sharing a cache dir at once).
            #
            # Built with the VALIDATION kwargs, since validation is the split
            # that asks for the extra per-item payload; the training collator
            # then drops it (`shared_train_collate_fn`).
            full_dataset = train_builder.dataset(transform=val_transform, **val_kwargs)
            if debug_share_train_val:
                self.log.warning(
                    "dataloaders(debug_share_train_val=True): train and val share "
                    "the same data -- diagnostic only, not a real split."
                )
                train_dataset = val_dataset = full_dataset
            else:
                # Split at BLOCK granularity (whole contiguous chunks of
                # block_size indices), not per-sample: a per-sample-random
                # split scatters both halves across the whole table, which
                # defeats shard-cache locality just as badly as no split at
                # all. Block order is shuffled once (seeded by dataset_seed)
                # before blocks are dealt out, so the two splits stay
                # interleaved across the whole table rather than being one
                # contiguous chunk each -- only the split BOUNDARIES move to
                # block granularity.
                #
                # `None` for val's fraction means "everything else", which is
                # exact: two independently-rounded fractions summing to 1.0 can
                # leave a block or two assigned to neither split.
                n = len(full_dataset)
                train_idx, val_idx = self.split_indices(
                    n, block_size, [var.train_val_split, None], seed=var.dataset_seed,
                )
                train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
                val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
            train_collate_fn = self._module_hook(
                model, 'shared_train_collate_fn',
                self._module_hook(model, 'train_collate_fn'),
            )

        # The val loader reads at most `val_max_batches` batches, so sizing it
        # like the train loader fetches -- and discards -- an order of
        # magnitude more shards than it reads, evicting the train working set
        # on every validation. See val_loader_workers().
        val_workers, val_prefetch = val_loader_workers(
            self.num_workers, self.prefetch_factor, var.val_max_batches,
        )
        if val_workers != self.num_workers:
            self.log.info(
                "val loader sized for val_max_batches=%s: num_workers=%d "
                "prefetch_factor=%d (train: %d/%d)",
                var.val_max_batches, val_workers, val_prefetch,
                self.num_workers, self.prefetch_factor,
            )

        # A sampler in place of shuffle=True: each Subset above was built by
        # concatenating whole blocks in order, so its own index space is still
        # block-structured -- chunking it by block_size (with no further
        # knowledge of the original global boundaries) keeps every access
        # localized to a handful of shards instead of scattering across the
        # whole table. Re-shuffled per epoch via set_epoch(), which the Trainer
        # calls automatically.
        train_sampler = self.Sampler(
            len(train_dataset), block_size, seed=var.dataset_seed,
        )
        val_sampler = None
        if var.val_shuffle:
            # fixed_epoch=True: the same val permutation for the life of the
            # run, so val_max_batches always picks out the same held-out
            # subset. Deliberately unlike the train sampler above, which
            # *should* reshuffle every epoch.
            val_sampler = self.Sampler(
                len(val_dataset), block_size, seed=var.val_seed, fixed_epoch=True,
            )

        training_dataloader = self._dataloader(
            train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            collate_fn=train_collate_fn,
            sampler=train_sampler,
            loader_cls=self.Loader,
        )
        val_dataloader = self._dataloader(
            val_dataset,
            batch_size=self.val_batch_size,
            num_workers=val_workers,
            prefetch_factor=val_prefetch,
            collate_fn=self._module_hook(model, 'val_collate_fn'),
            sampler=val_sampler,
        )
        return training_dataloader, val_dataloader

    @property
    def train_batch_size(self) -> int:
        """Batch size for the training loader."""
        return self.var.batch_size

    @property
    def val_batch_size(self) -> int:
        """Batch size for the validation loader.

        Separate from :attr:`train_batch_size` so a subclass whose validation
        metric depends on its own batch size -- an in-batch retrieval score,
        where chance is ``1/B`` -- can vary one without the other.
        """
        return self.var.batch_size

    # 8. Training banner ────────────────────────────────────────────
    #
    # Split out of __build__ so a subclass with a different architecture can
    # restate the rows without duplicating the training loop around them.

    @staticmethod
    def _banner_trunc(s, w):
        s = str(s)
        return s if len(s) <= w else s[:w - 3] + "..."

    @staticmethod
    def _fmt_params(n):
        if n >= 1e9: return f"{n / 1e9:.1f}B"
        if n >= 1e6: return f"{n / 1e6:.1f}M"
        if n >= 1e3: return f"{n / 1e3:.1f}K"
        return str(n)

    def _banner_param_str(self, model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return f"{self._fmt_params(total)} (trainable: {self._fmt_params(trainable)})"

    def _banner_dataset_tag(self):
        return getattr(self.var.training_dataset_builder, 'tag', None) \
            or type(self.var.training_dataset_builder).__name__

    def _banner_rows(self, model, ckpt, ckpts_dir):
        """``[(label, value), ...]`` for the banner.  Extend in a subclass."""
        var = self.var
        _W = self.BANNER_WIDTH
        return [
            ("PARAMS",    self._banner_param_str(model)),
            ("DATASET",   str(self._banner_dataset_tag())),
            ("EPOCHS",    str(var.max_epochs)),
            ("BATCH",     f"{var.batch_size} x {var.accumulate_grad_batches} accum"
                          f" = {var.batch_size * var.accumulate_grad_batches} eff."),
            ("PRECISION", str(var.precision)),
            # RESUME, not CKPT: this is the optimizer/step state the run
            # continues from, which is a different question from where the
            # weights came from. "CKPT None" read as "no weights loaded".
            ("RESUME",    self._banner_trunc(
                ckpt if ckpt else "none - fresh run at step 0", _W - 14)),
            ("LOCALWORK", self._banner_trunc(ckpts_dir, _W - 14)),
            ("KEY",       self._banner_trunc(self.key, _W - 14)),
        ]

    def _print_training_banner(self, model, ckpt, ckpts_dir):
        _W = self.BANNER_WIDTH
        _border = "╔" + "═" * _W + "╗"
        _footer = "╚" + "═" * _W + "╝"
        _sep    = "╠" + "═" * _W + "╣"
        # An emoji in the title is 2 columns wide, so .center() lands one
        # column off per emoji; BANNER_TITLE_EMOJI says how many to compensate.
        _title_pad = self.BANNER_TITLE.center(_W - self.BANNER_TITLE_EMOJI)
        print(_border)
        print("║" + _title_pad + "║")
        print(_sep)
        for _label, _val in self._banner_rows(model, ckpt, ckpts_dir):
            _line = f"  {_label:<10}{_val}"
            print("║" + self._banner_trunc(_line, _W).ljust(_W) + "║")
        print(_footer)

    # 9. Trainer assembly ───────────────────────────────────────────

    def trainer_kwargs(self, *, ckpts_dir, callbacks, tb_logger):
        """The ``L.pytorch.Trainer`` keyword arguments for this run."""
        var = self.var
        # Lightning's `val_check_interval` counts MICRO-BATCHES
        # (`_should_check_val_fx` tests `(total_batch_idx + 1) %
        # val_check_batch`), but every other *_n_steps knob here means
        # OPTIMIZER STEPS: `ckpt_every_n_steps` tests `trainer.global_step`
        # and `train_log_every_n_steps` tests `_batches_that_stepped`, both of
        # which advance once per optimizer step. So the raw value has to be
        # scaled, or `val_every_n_steps=5` with `accumulate_grad_batches=2`
        # validates every 2.5 steps and the TensorBoard x-axis -- which is
        # `_batches_that_stepped` -- shows val points at 2, 4, 7, 9, 12, 14
        # instead of 5, 10, 15.
        val_check_interval = var.val_every_n_steps * var.accumulate_grad_batches
        if var.accumulate_grad_batches > 1:
            self.log.info(
                "val_every_n_steps=%d optimizer steps -> val_check_interval=%d "
                "micro-batches (accumulate_grad_batches=%d)",
                var.val_every_n_steps, val_check_interval, var.accumulate_grad_batches,
            )
        kwargs = dict(
            default_root_dir=ckpts_dir,
            max_epochs=var.max_epochs,
            log_every_n_steps=var.train_log_every_n_steps,
            val_check_interval=val_check_interval,
            check_val_every_n_epoch=None,
            callbacks=callbacks,
            devices=self.n_devices,
            logger=tb_logger,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            # Lightning's own ModelCheckpoint is off: the callbacks below
            # upload each checkpoint and delete the local copy immediately,
            # which is what keeps local disk bounded over a long run.
            enable_checkpointing=False,
        )
        if var.max_training_steps is not None:
            kwargs['limit_train_batches'] = var.max_training_steps
        if var.val_max_batches is not None:
            kwargs['limit_val_batches'] = var.val_max_batches
        if var.gradient_clip_val > 0.0:
            kwargs['gradient_clip_val'] = var.gradient_clip_val
            kwargs['gradient_clip_algorithm'] = var.gradient_clip_algorithm
        if var.precision is not None:
            kwargs['precision'] = var.precision
        if var.accumulate_grad_batches > 1:
            kwargs['accumulate_grad_batches'] = var.accumulate_grad_batches
        return kwargs

    def callbacks(self, *, ckpts_dir):
        """The callbacks for this run: identity logging and the checkpointers."""
        _still = self
        _ckpts_dir = ckpts_dir
        callbacks = []

        class _LogCiteOnStart(L.pytorch.Callback):
            """Record the block's reconstructible identity in TensorBoard.

            So that a run directory answers "what produced this curve?" on its
            own -- `cite()` is the text that rebuilds this exact block. Wrapped
            in a try/except because a logging nicety must never be the thing
            that kills a training run.
            """

            def on_train_start(self, trainer, pl_module):
                if trainer.global_rank != 0 or trainer.logger is None:
                    return
                try:
                    trainer.logger.experiment.add_text(
                        "cite", f"```\n{_still.cite()}\n```",
                        global_step=trainer.global_step,
                    )
                except Exception as e:
                    _still.log.warning("could not log cite() to TensorBoard: %s", e)

        def _upload_and_free(local_path, filename):
            if _still.is_local_fs:
                return
            try:
                remote_path = os.path.join(_still.dirpath("ckpts", ensure=True), filename)
                _still.push(local_path, remote_path, free_src=True, show_progress=True)
                _still.log.info("Uploaded %s and freed local copy", filename)
            except Exception as e:
                _still.log.warning("Ckpt upload failed: %s", e)

        def _journal_ckpt(ckpt_tag):
            try:
                _still.write_journal_entry(event=f"ckpt={ckpt_tag}")
            except Exception as e:
                _still.log.warning("Ckpt journal entry failed: %s", e)

        def _save(trainer, epoch, step):
            filename = f"epoch={epoch:03d}-step={step:07d}.ckpt"
            local_path = os.path.join(_ckpts_dir, filename)
            trainer.save_checkpoint(local_path)
            _still.log.info("Saved checkpoint: %s", filename)
            _upload_and_free(local_path, filename)
            _journal_ckpt(filename)

        callbacks.append(_LogCiteOnStart())

        if self.var.ckpt_every_n_steps is not None:
            _every_n_steps = self.var.ckpt_every_n_steps

            class _StepCheckpoint(L.pytorch.callbacks.Callback):
                # `on_train_batch_end` fires once per MICRO-batch, but
                # `trainer.global_step` only advances once per real optimizer
                # step -- under accumulate_grad_batches > 1 several
                # micro-batches share one global_step. Without the dedup guard,
                # every micro-batch in that window would re-save and re-push
                # the SAME checkpoint, each a blocking remote upload sitting
                # directly in the training loop.
                _last_ckpt_step = -1

                def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                    step = trainer.global_step
                    if step == 0 or step % _every_n_steps != 0 or step == self._last_ckpt_step:
                        return
                    self._last_ckpt_step = step
                    _save(trainer, trainer.current_epoch, step)

            callbacks.append(_StepCheckpoint())

        if self.var.ckpt_every_n_epochs is not None:
            _every_n_epochs = self.var.ckpt_every_n_epochs

            class _EpochCheckpoint(L.pytorch.callbacks.Callback):
                def on_train_epoch_end(self, trainer, pl_module):
                    epoch = trainer.current_epoch
                    if (epoch + 1) % _every_n_epochs != 0:
                        return
                    _save(trainer, epoch, trainer.global_step)

            callbacks.append(_EpochCheckpoint())

        return callbacks

    # 10. Resume ────────────────────────────────────────────────────

    def _resume_plan(self):
        """The checkpoint this run *intends* to resume, named without downloading.

        The banner reports this, and the banner is printed before the fetch --
        because the banner is what lets you sanity-check the configuration, and
        the fetch is a multi-GB download.  Announcing the run only after
        committing to the transfer is backwards.

        So it can only be an intention: naming the checkpoint is a cheap remote
        listing, whereas resolving it downloads and validates, falling back to
        the next-latest if the newest is corrupt.  Any divergence is logged
        explicitly afterwards, so the banner cannot quietly lie.
        """
        plan = None
        if not self.var.from_scratch:
            plan = self._find_latest_ckpt_remote()
        if plan is None and self.var.ckpt is not None:
            plan = (self.var.ckpt.key if hasattr(self.var.ckpt, 'key')
                    else str(self.var.ckpt))
            # Says which of the two things the banner's RESUME row means. They
            # are not the same run: continuing this one keeps its optimizer and
            # step count, starting from another one keeps only its weights.
            if self._warm_start_block() is not None:
                plan = f"warm start, weights only - {plan}"
        return plan

    def _warm_start_block(self):
        """``VAR.ckpt`` when it is another *block* to start from, else None.

        A block is asked for its own latest checkpoint; a bare path is taken as
        given.  The distinction decides how the checkpoint is loaded -- see
        :meth:`_resolve_resume_ckpt`.
        """
        initial = self.var.ckpt
        return initial if hasattr(initial, 'find_latest_ckpt') else None

    def _resolve_resume_ckpt(self):
        """``(local_ckpt_or_None, own_it, warm_start)`` -- what to start from.

        ``own_it`` is True only for our *own* latest checkpoint: that is always
        a local path under the staging ``ckpts`` dir which is also safely
        duplicated on remote (that is how it got there -- downloaded from
        remote, or already uploaded by the save callback), so it is safe to
        free once loaded.  Never True for a ``var.ckpt``-derived path: that may
        point at another block's cache or an arbitrary user-supplied file,
        neither of which we own or know to be remote-backed.

        ``warm_start`` is True when the checkpoint came from *another block*
        (:meth:`_warm_start_block`), which decides how it is loaded: weights
        only, at step 0, with a fresh optimizer.  Another run's optimizer
        moments and step counter say nothing about this one, and carrying them
        over is worse than useless -- a source that already reached its own
        ``max_epochs`` leaves this run with nothing left to do, so it "trains"
        instantly and writes a ``_COMPLETE`` marker over an untrained model.

        A bare *path* in ``var.ckpt`` keeps the full-resume behaviour: it is
        how you point a run at its own checkpoint that has moved, where the
        optimizer state is exactly what you want back.  So the escape hatch
        from the weights-only default is to pass the path rather than the
        block.
        """
        ckpt, own_it, warm_start = None, False, False
        if not self.var.from_scratch:
            ckpt = self.find_latest_ckpt(pull=True)
            own_it = ckpt is not None
        # Our own run always wins: once this still has checkpoints of its own,
        # a resume continues it and the warm-start source is history.
        if ckpt is None and self.var.ckpt is not None:
            initial = self._warm_start_block()
            if initial is not None:
                ckpt = initial.find_latest_ckpt(pull=True)
                warm_start = True
                if ckpt is None:
                    raise FileNotFoundError(
                        f"the warm-start block in VAR.ckpt has no checkpoints: {initial}"
                    )
            else:
                ckpt = str(self.var.ckpt)
        return ckpt, own_it, warm_start

    def _load_weights_only(self, model, ckpt, *, why):
        """Load *ckpt*'s weights into *model*, leaving the run at step 0.

        No optimizer state, no epoch or step counter: the caller wants the
        weights and a fresh training schedule.  *why* names the reason in the
        log line, since the two callers are quite different situations.

        ``strict=False``, not ``self.strict_loading``: a weights-only load is
        exactly the case where the two state dicts are expected not to line up
        exactly (a warm start into a changed head), and every existing still
        relies on that.  ``strict_loading`` is accepted by the constructor and
        wired to nothing -- see its note there.

        What ``strict=False`` will not catch is a checkpoint whose parameter
        *names* have nothing to do with this model -- a warm start pointed at
        the wrong architecture entirely.  That loads nothing at all, silently,
        and the run then trains from random init while its banner says
        otherwise, so it is refused here instead.
        """
        self.log.info("Loading weights only (%s) from %s", why, ckpt)
        checkpoint = torch.load(ckpt, weights_only=False)
        if "state_dict" not in checkpoint:
            raise KeyError(
                f"{ckpt} has no 'state_dict' to warm-start from "
                f"(top-level keys: {sorted(checkpoint)})"
            )
        state_dict = checkpoint["state_dict"]
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        loaded = len(state_dict) - len(unexpected)
        if loaded == 0:
            raise ValueError(
                f"nothing loaded from {ckpt}: none of its {len(state_dict)} "
                f"parameter names match this model. It is a checkpoint of a "
                f"different architecture -- warm-starting from it would train "
                f"from random init while reporting otherwise."
            )
        self.log.info(
            "Loaded %d/%d tensors (missing=%d unexpected=%d)",
            loaded, len(state_dict), len(missing), len(unexpected),
        )

    def _report_resume_divergence(self, ckpt, resume_plan, *, warm_start=False):
        """Say so, loudly, when the fetch did not land where the banner said.

        Skipped for a warm start, where the plan names the *block* rather than
        a checkpoint file: comparing a key against a filename would report a
        divergence on every warm start there has ever been.
        """
        if (not warm_start and ckpt is not None and resume_plan is not None
                and os.path.basename(str(ckpt)) != os.path.basename(str(resume_plan))):
            self.log.warning(
                "resume checkpoint is NOT the one in the banner: %s -> %s "
                "(the newer one failed validation and was skipped)",
                os.path.basename(str(resume_plan)), os.path.basename(str(ckpt)),
            )
        elif ckpt is None and resume_plan is not None:
            self.log.warning(
                "banner announced a resume from %s but none could be fetched; "
                "starting at step 0 instead", resume_plan,
            )

    # 11. Build (training) ──────────────────────────────────────────

    def __build__(self):
        """Run the training loop."""
        logs_dir = self._local_logs_dir
        ckpts_dir = self._local_ckpts_dir
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(ckpts_dir, exist_ok=True)

        # Registered before anything can fail: a run killed by SIGINT, an OOM,
        # or a preempted node has checkpoints on local disk that are worth more
        # than the exception, and nothing else would ever move them.
        def _atexit_sync():
            self.log.info("atexit: syncing checkpoints to remote...")
            self._sync_to_remote(reason='atexit')

        atexit.register(_atexit_sync)
        self.linklogs()

        tb_logger = L.pytorch.loggers.TensorBoardLogger(
            save_dir=logs_dir,
            default_hp_metric=False,
            name="",
            version=f"run_{datetime.now().strftime('%Y-%m-%d_%H.%M.%S')}",
        )

        self.log.info("path/ckpts: %s", self.dirpath('ckpts'))
        self.log.info("path/logs: %s", self.dirpath('logs'))

        callbacks = self.callbacks(ckpts_dir=ckpts_dir)
        trainer = L.pytorch.Trainer(**self.trainer_kwargs(
            ckpts_dir=ckpts_dir, callbacks=callbacks, tb_logger=tb_logger,
        ))

        original_matmul_precision = torch.get_float32_matmul_precision()
        if self.var.matmul_precision is not None:
            torch.set_float32_matmul_precision(self.var.matmul_precision)

        try:
            L.seed_everything(self.var.still_seed, workers=True)
            model = self.lightning_module

            resume_plan = self._resume_plan()
            self._print_training_banner(model, resume_plan, ckpts_dir)

            ckpt, own_resume_ckpt, warm_start = self._resolve_resume_ckpt()
            self._report_resume_divergence(ckpt, resume_plan, warm_start=warm_start)

            fit_kwargs = {}
            if ckpt is not None:
                if warm_start or self.var.reset_optimizer_state:
                    self._load_weights_only(
                        model, ckpt,
                        why=("warm start from another block" if warm_start
                             else "optimizer reset"),
                    )
                    # Fully read by the load above -- free it now rather than
                    # leaving it on disk for the run's duration.
                    if own_resume_ckpt:
                        self._free_local_ckpt(ckpt)
                else:
                    self.log.info("Resuming from checkpoint: %s", ckpt)
                    fit_kwargs["ckpt_path"] = ckpt
                    if own_resume_ckpt:
                        # Lightning reads ckpt_path lazily inside fit(), so it
                        # cannot be freed here -- it is not loaded yet. Freed
                        # from on_train_start instead, which Lightning fires
                        # only once checkpoint restore is fully complete.
                        _resume_ckpt_path = ckpt
                        _still = self

                        class _FreeResumeCkpt(L.pytorch.Callback):
                            def on_train_start(self, trainer, pl_module):
                                _still._free_local_ckpt(_resume_ckpt_path)

                        trainer.callbacks.append(_FreeResumeCkpt())
            else:
                # This branch is about RUN CONTINUITY -- whether there is
                # optimizer/step state to resume -- NOT about weight init.
                # Weights were already set when the module was built above:
                # random, pre-trained, or warm-started from another still.
                # "from scratch" read as "randomly initialised" and
                # contradicted whatever the module logged seconds earlier.
                self.log.info(
                    "No resume checkpoint - starting a fresh run at step 0 "
                    "(weight initialisation is unaffected; see the banner)"
                )

            self.log.info("Building dataloaders...")
            training_dataloader, val_dataloader = self.dataloaders(model=model)
            self.log.info("DataLoaders ready - starting trainer.fit()")

            trainer.fit(
                model=model,
                train_dataloaders=training_dataloader,
                val_dataloaders=val_dataloader,
                **fit_kwargs,
            )

            self.UNSAFE_complete(OVERRIDE=True)
            self._sync_to_remote(reason='post-fit')
        finally:
            # Unregistered first: the sync below does the same work, and an
            # atexit handler that runs afterwards would repeat a multi-GB push.
            atexit.unregister(_atexit_sync)
            self._sync_to_remote(reason='finally')
            torch.set_float32_matmul_precision(original_matmul_precision)

        return self

    # 12. Export ────────────────────────────────────────────────────

    #: Marker written into every generated module's docstring.  :meth:`export`
    #: refuses to overwrite a file that lacks it, so a hand-written module
    #: cannot be clobbered by a stray path argument.
    EXPORT_MARKER = "AUTO-GENERATED by Still.export()"

    #: Markers written under earlier names.  Recognised for the overwrite check
    #: only, so a module generated before a rename stays regenerable rather
    #: than raising FileExistsError.
    LEGACY_EXPORT_MARKERS = ()

    def export(self, path=None, *, ckpt=None, overwrite=True, validate=False):
        """Generate a standalone module that rebuilds this still's model.

        Returns the source, and writes it to *path* when given.  The generated
        module depends on the ``Model``/``Lightning`` classes' own modules and
        on nothing else from here: no dbx, no Datablock identity, no
        ``build_tree()``.  What it carries is *configuration* -- the ``cfg_``
        arguments as literals, plus the checkpoint path -- so it reaches the
        weights without reconstructing this block, and no changed ``VAR``
        default can silently point it at a different run.

        Only available in cfg mode.  In explicit mode the module's
        construction lives in a ``LightningBuilder`` and there is no general way
        to restate it, so a subclass that wants an export writes its own (see
        ``IJEPAsaurUSStill.export`` in soundworld for one that splices in
        vendored source).

        Parameters
        ----------
        path : str, optional
            Where to write.  Not written when omitted.
        ckpt : str, optional
            The checkpoint to bake in as the default.  Defaults to this
            still's own latest.  The generated module still takes one per call.
        overwrite : bool, default True
            Whether to replace an existing file.  Only ever applies to a file
            carrying :attr:`EXPORT_MARKER` -- overwriting anything else raises
            regardless, since it was not ours to replace.
        validate : bool, default False
            Whether resolving the default checkpoint should download and
            validate it.  Off by default: generating source should not pull a
            multi-GB file.
        """
        if self.Model is None or self.Lightning is None:
            raise NotImplementedError(
                f"{type(self).__name__}.export() needs cfg mode (Model and Lightning "
                f"class attributes) to know how to restate the model's construction. "
                f"Override export() for an explicit-mode still"
            )
        if ckpt is None:
            ckpt = self.find_latest_ckpt(pull=validate)
            if ckpt is None:
                self.log.warning(
                    "export: no checkpoint under %s -- generating with CKPT=None; "
                    "the module will need one passed in", self.dirpath('ckpts'),
                )
        source = self._export_source(ckpt=ckpt)
        if path is not None:
            self._write_export(source, path, overwrite=overwrite)
            self.log.info("export: wrote %s (ckpt=%s)", path, ckpt)
        return source

    def _export_source(self, *, ckpt):
        """The generated module's source.  Override to change what it exposes."""
        # One import per module, so Model and Lightning sharing a module
        # produce `from m import Model, Lightning` rather than two lines
        # importing from the same place.
        by_module: dict = {}
        for cls in (self.Model, self.Lightning):
            by_module.setdefault(cls.__module__, []).append(cls.__name__)
        imports = [f"from {mod} import {', '.join(names)}"
                   for mod, names in by_module.items()]
        lines = [
            '"""',
            f"{self.EXPORT_MARKER}",
            '',
            f"Rebuilds the model trained by {type(self).__name__}, tag {self.tag!r}.",
            '',
            "Depends only on the model's own modules -- no dbx, no Datablock",
            "identity. The configuration below is what that run actually used.",
            '"""',
            '',
            'import torch',
            '',
            *imports,
            '',
            f"CKPT = {ckpt!r}",
            '',
            f"MODEL_CFG = {self._export_cfg_repr(self.model_cfg)}",
            '',
            f"LIGHTNING_CFG = {self._export_cfg_repr(self.lightning_cfg)}",
            '',
            '',
            'def build_model(ckpt=CKPT, *, device="cpu", strict=False):',
            '    """Instantiate the model and load *ckpt* into it."""',
            f"    model = {self.Model.__name__}(**MODEL_CFG)",
            f"    module = {self.Lightning.__name__}(model, **LIGHTNING_CFG)",
            '    if ckpt is not None:',
            '        state = torch.load(ckpt, map_location=device, weights_only=False)',
            '        module.load_state_dict(state.get("state_dict", state), strict=strict)',
            '    return module.to(device).eval()',
            '',
            '',
            'if __name__ == "__main__":',
            '    m = build_model()',
            '    n = sum(p.numel() for p in m.parameters())',
            '    print(f"built {type(m).__name__}: {n:,} parameters, ckpt={CKPT}")',
            '',
        ]
        return '\n'.join(lines)

    @staticmethod
    def _export_cfg_repr(cfg):
        """A cfg dict as readable, one-per-line source."""
        if not cfg:
            return '{}'
        body = ''.join(f"    {k!r}: {v!r},\n" for k, v in sorted(cfg.items()))
        return '{\n' + body + '}'

    def _write_export(self, source, path, *, overwrite=True):
        """Write *source* to *path*, refusing to clobber a file that is not ours."""
        if os.path.exists(path):
            with open(path, 'r') as f:
                existing = f.read(4096)
            markers = (self.EXPORT_MARKER, *self.LEGACY_EXPORT_MARKERS)
            if not any(m in existing for m in markers):
                raise FileExistsError(
                    f"{path} exists and was not generated by export() (no "
                    f"{self.EXPORT_MARKER!r} marker); refusing to overwrite it "
                    f"-- pick another path"
                )
            if not overwrite:
                raise FileExistsError(
                    f"{path} already exists; pass overwrite=True to replace it"
                )
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        with open(path, 'w') as f:
            f.write(source)
        return path


# ═══════════════════════════════════════════════════════════════════════
# scaffold_still — the VAR, written down
# ═══════════════════════════════════════════════════════════════════════

def scaffold_still(Model, Lightning, *, name, version=1, base='Still',
                   entrypoint=None, docstring=None):
    """The source of a :class:`Still` subclass mirroring the ``cfg_`` surface.

    Deterministic code generation, run once and the result committed -- not
    reflection at import time.  The difference matters because every VAR field
    is hashed into the storage key (see the module docstring): as source, a new
    knob is a diff that re-keys the block visibly, and as reflection it is an
    edit in a file that does not mention dbx silently orphaning every artifact
    the old key addressed.

    Parameters
    ----------
    Model, Lightning : type
        The classes to mirror.  Their ``cfg_``-prefixed keyword arguments
        become VAR fields, in declaration order, Model's first.
    name : str
        Class name for the generated still.
    version : int, default 1
        Its ``VERSION``.
    base : str, default ``'Still'``
        Base class name, as it will be spelled in the generated source.
    entrypoint : str, optional
        When given, also emit a pipeline function of that name taking
        ``var_``-prefixed arguments -- the convention that keeps
        identity-affecting arguments visually distinct from operational ones at
        the call site.
    docstring : str, optional
        Class docstring.  A generated one naming both classes by default.

    Returns
    -------
    str
        Python source.  Write it next to the model, read it, commit it.
    """
    params = cfg_params({'model': Model, 'lightning': Lightning})
    reserved = sorted(n for n in params if n in Still.RESERVED_VAR_NAMES)
    if reserved:
        raise ValueError(
            f"cannot mirror {[CFG_PREFIX + n for n in reserved]}: {base} already "
            f"defines VAR field(s) {reserved} with a different meaning, and one "
            f"field cannot carry both. Rename the {CFG_PREFIX} argument(s) on "
            f"{Model.__name__}/{Lightning.__name__}"
        )
    if not params:
        raise ValueError(
            f"neither {Model.__name__} nor {Lightning.__name__} declares any "
            f"{CFG_PREFIX} keyword argument, so there is nothing to mirror. "
            f"Prefix the arguments you want exposed as VAR fields with "
            f"{CFG_PREFIX!r}"
        )

    doc = docstring or (
        f"Training run for :class:`{Model.__name__}` via "
        f":class:`{Lightning.__name__}`.\n\n"
        f"    Generated by ``dbx.stills.scaffold_still``. The VAR fields below "
        f"mirror\n    the ``{CFG_PREFIX}`` arguments of those two classes; every one of them "
        f"is\n    hashed into this block's storage key, so adding, removing or "
        f"re-defaulting\n    one re-keys every run. Regenerate rather than hand-editing, "
        f"and read the\n    diff."
    )

    lines = [
        f"# Generated by dbx.stills.scaffold_still({Model.__name__}, "
        f"{Lightning.__name__}).",
        "# Needs, in the module this lands in:",
        "#     from dataclasses import dataclass, field",
        f"#     from dbx.stills import {base}",
        f"#     from <your module> import {Model.__name__}, {Lightning.__name__}",
        '',
        '',
        f"class {name}({base}):",
        f'    """{doc}"""',
        '',
        f"    VERSION = {version}",
        f"    Model = {Model.__name__}",
        f"    Lightning = {Lightning.__name__}",
        '',
        '    @dataclass',
        f"    class VAR({base}.VAR):",
    ]
    for p in params.values():
        roles = '/'.join(p.roles)
        lines.append(f"        {p.name}: {p.annotation_text()} = {_default_source(p)}"
                     f"  # {roles}")
    lines.append('')

    if entrypoint:
        lines.extend(_entrypoint_source(entrypoint, name, params))

    return '\n'.join(lines) + '\n'


def _default_source(param: Cfgparam) -> str:
    """A cfg default as dataclass-field source.

    A mutable default needs ``field(default_factory=...)``, which is a
    dataclass rule rather than anything to do with dbx: a shared list default
    would be mutated by whichever instance touched it first.
    """
    default = param.default
    if default is inspect.Parameter.empty:
        return 'None  # REQUIRED upstream: no default to mirror'
    if isinstance(default, (list, dict, set)):
        return f"field(default_factory=lambda: {default!r})"
    return repr(default)


def _entrypoint_source(fname: str, clsname: str, params: dict) -> list:
    """A pipeline entrypoint taking ``var_`` arguments, as source lines."""
    lines = [
        '',
        f"def {fname}(",
        '    *,',
        '    url=None,',
        '    storage_options=None,',
        '    local=None,',
        '    tag=None,',
        '    training_dataset_builder=None,',
        '    validation_dataset_builder=None,',
    ]
    for p in params.values():
        lines.append(f"    var_{p.name}={_default_source(p).split('  #')[0]},")
    lines.extend([
        '    # Operational: no var_ prefix, so no effect on identity.',
        '    n_devices=1,',
        '    tensorlogs_root=None,',
        '    num_workers=4,',
        '    prefetch_factor=4,',
        "):",
        f'    """Build a :class:`{clsname}`.',
        '',
        '    Every ``var_`` argument affects the identity hash, and so the storage',
        '    key. Every other argument is operational and does not.',
        '    """',
        f"    still = {clsname}(",
        '        url=url, storage_options=storage_options, local=local,',
        '        spec=dict(',
        '            training_dataset_builder=training_dataset_builder,',
        '            validation_dataset_builder=validation_dataset_builder,',
    ])
    for p in params.values():
        lines.append(f"            {p.name}=var_{p.name},")
    lines.extend([
        '        ),',
        '        n_devices=n_devices,',
        '        tensorlogs_root=tensorlogs_root,',
        '        num_workers=num_workers,',
        '        prefetch_factor=prefetch_factor,',
        '    )',
        '    return still if tag is None else still.set(tag=tag)',
        '',
    ])
    return lines


# ═══════════════════════════════════════════════════════════════════════
#  The names these classes used to have
# ═══════════════════════════════════════════════════════════════════════

#: This file was ``dbx/datastills.py``. It gets no module alias, unlike
#: :mod:`dbx.backbones` and :mod:`dbx.probes`: it is a week old, has never been
#: released, and no artifact anywhere is stored under a ``dbx.datastills.*``
#: anchor -- and a recorded string that resolves is the only thing a module
#: alias buys.
#:
#: The class names are aliased for source that already imports them.
Datastill = Still
Datalightning = LightningBuilder
Dataweights = Weights
