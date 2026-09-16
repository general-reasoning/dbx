"""dbx.stills — ``Still``, the training-run Datablock.

A *still* is one training run, addressed by its configuration: its weights and
its TensorBoard logs live under a key derived from the hash of everything that
produced them.  Every such run in this codebase had grown its own copy of the
same ~700 lines -- the ``done`` topic, the checkpoint
save/upload/free/resume dance, the log symlink, the atexit sync, the
``UNSAFE_*`` helpers and the Lightning ``Trainer`` assembly.  This module holds
one of each.

Not imported by ``dbx/__init__.py``
-----------------------------------
``import dbx`` must work without torch, and this module needs both torch and
lightning at module scope.  Import it by name -- ``from dbx.stills import
Still`` -- exactly as ``dbx.datastreams`` is imported.

Everything a still trains with is a block
-----------------------------------------
A ``Still`` constructs nothing itself.  Each collaborator arrives as another
Datablock in VAR, and the still asks it for the one thing it builds::

    VAR.model_builder                 ModelBuilder        .model()
    VAR.lightning_builder             LightningBuilder    .lightning_module
    VAR.training_dataset_builder      DatasetBuilder      .dataset(transform=)
    VAR.validation_dataset_builder    DatasetBuilder      .dataset(transform=)
    VAR.ckpt_builder                  CheckpointBuilder   .find_latest_ckpt()

The first four are required: a still missing any of them refuses to construct,
naming what is absent.  ``ckpt_builder`` is optional and says which earlier run
to warm-start from.

Why builders and not configuration knobs
----------------------------------------
Because a Datablock carries a hash and states its own configuration in VAR,
and a loose constructor argument does neither.  The architecture a run trained
is part of what that run IS.  Held as a builder block it has an identity the
still's own key is computed from, and a reader of the journal can follow it
back to the thing that produced it.  Held as keyword arguments it is invisible:
two runs of genuinely different models can share a key, and no recorded
signature says which model either of them was.

This supersedes the ``cfg_`` protocol, in which a Still mirrored its Model's
and LightningModule's ``cfg_``-prefixed constructor arguments into its own VAR
and checked on every construction that the two had not drifted apart.  Builders
make the mirroring unnecessary -- each class's configuration lives in the VAR
of the block that builds it, hashed once and named once -- so the mirroring,
the drift check, and the ``scaffold_still`` generator that wrote the mirror are
all gone.  ``Still.VERSION`` is 2 as of that change; a VERSION 1 still cannot
address what a VERSION 2 still writes.

The identity-affecting half of ``VAR`` is the builders plus the training
configuration (``max_epochs``, ``batch_size``, ``precision``, the ``val_*``
knobs, ...).  Everything operational -- device count, worker count, where
TensorBoard logs get symlinked -- is a constructor keyword and affects no hash.
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
from typing import Any

import fsspec
import torch

import lightning as L
import lightning.pytorch.callbacks
import lightning.pytorch.loggers

from dbx.datablocks import DIR, Datablock
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
    'CheckpointBuilder',
    'CheckpointPath',
    'DatasetBuilder',
    'LightningBuilder',
    'ModelBuilder',
    'Still',
    'Weights',
    # Re-exported from dbx.datastreams, which is torch-only: importing this
    # module pulls in lightning, and the split/sizing helpers are useful to
    # code that has no business doing that.
    'block_split_indices',
    'block_split_ranges',
    'shuffled_block_order',
    'val_loader_workers',
]


# ═══════════════════════════════════════════════════════════════════════
# Checkpoint files on disk
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


# ═══════════════════════════════════════════════════════════════════════
# What a Still's VAR accepts — the Datablocks it delegates to
# ═══════════════════════════════════════════════════════════════════════
#
# Nominal, not structural: these are abstract Datablocks, and a collaborator
# has to INHERIT one rather than merely carry the right method. That is the
# point. What a run trained on is part of what the run IS, so a dataset builder
# or a warm-start source has to carry a hash and state its own configuration in
# VAR -- which is exactly what a duck-typed object cannot do, leaving a still's
# identity describing an upstream it has no way to name.
#
# Neither class declares VERSION or TOPICS, deliberately. Datablock carries no
# VERSION of its own -- identity reads it as None when the attribute is absent
# -- so declaring one here would move the hash of every existing builder the
# day it starts inheriting from this class. Same reasoning as LightningBuilder's
# missing TOPICS below. A subclass declares both for itself.
#
# Every VAR field annotated with one of these also admits `str`, because dbx
# accepts a SPECLINE anywhere a block can go and resolves it later. Keep the
# `str` when narrowing one of these further: Datablock._coerce_to_annotation()
# reads the annotation as TEXT and declines to coerce when it mentions `str`,
# so dropping it arms literal-coercion on a field that holds paths and
# speclines -- and a coerced value renders, and therefore hashes, differently.
#
# The third thing a Still's VAR takes -- the LightningModule to train in
# explicit mode -- is LightningBuilder itself, defined just below. It needs no
# separate declaration here: it was always the block VAR.lightning_builder holds.


class DatasetBuilder(Datablock):
    """The block that builds the dataset a still trains or validates on.

    Subclass it and implement ``dataset()``.  ``transform`` is the model's own
    preprocessing, passed through from the LightningModule's
    ``train_transform`` / ``val_transform`` hooks.  A builder carrying a
    ``tag`` has it reported in the training banner.
    """

    def dataset(self, *, transform=None, **kwargs) -> torch.utils.data.Dataset:
        raise NotImplementedError(
            f"{type(self).__name__}.dataset(): a DatasetBuilder builds the dataset "
            f"its VAR describes. Implement it, applying the transform it is given "
            f"-- the model's own preprocessing -- to the items it yields"
        )


class ModelBuilder(Datablock):
    """The block that builds the model a still trains.

    Subclass it and implement ``model()``.  The architecture and its
    hyperparameters live in this block's VAR, which is where they get hashed
    and named -- the still then depends on this block rather than restating
    any of it.
    """

    def model(self) -> torch.nn.Module:
        raise NotImplementedError(
            f"{type(self).__name__}.model(): a ModelBuilder builds the model its "
            f"VAR describes. Implement it, reading the architecture from self.var"
        )


class CheckpointBuilder(Datablock):
    """A block whose latest checkpoint another run can warm-start from.

    ``Still`` is the implementation, and ``VAR.ckpt`` holding one of these
    rather than a plain path is what marks a warm start.
    """

    def find_latest_ckpt(self, *, pull: bool = False) -> str | None:
        raise NotImplementedError(
            f"{type(self).__name__}.find_latest_ckpt(): a CheckpointBuilder names "
            f"the checkpoint another run may start from, or None when it has none"
        )


# ═══════════════════════════════════════════════════════════════════════
# LightningBuilder — the Datablock that owns a LightningModule
# ═══════════════════════════════════════════════════════════════════════

class LightningBuilder(Datablock):
    """A Datablock whose whole job is to build one ``LightningModule``.

    It writes nothing.  It exists so that a module's *construction* --
    architecture, optimiser schedule, loss configuration, and whichever
    upstream blocks supply its initial weights -- has an identity of its own,
    which the still that trains it then depends on.  Subclass it and implement
    ``__lightning_module__``.

    Subclasses must not override ``valid`` or ``__build__`` without
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
        ``__lightning_module__`` needs, and surfacing much later as a
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
          ``valid`` False and ``__pre_build__`` raises;
        * with ``validate_vars=False``, ``valid_var()`` answers True
          unconditionally, so ``valid`` is True and ``build()`` skips the
          block as already done.

        So this fires only for a subclass that overrides ``valid`` into
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
        so that a subclass can override ``__lightning_module__`` -- the
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
    ``path_local`` hands the consumer a **local path**, which is the
    only thing ``torch.load`` will take.

    Subclass it to teach it a naming scheme (a registry of published
    checkpoints, say) by overriding ``source_url``.
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
        ckpt: str | None = None   # registry key, url, local path, or None

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
        so the next ``path_local`` refetches rather than loading a
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

    def path_local(self) -> str | None:
        """A local path to the weights, downloading them if need be, or ``None``.

        ``None`` exactly when ``var.ckpt`` is ``None`` -- when the model is to
        be randomly initialised -- so a caller can pass the result straight
        through as a ``cfg_ckpt`` argument either way, with no branch of its
        own.

        ``torch.load`` needs a real file, which is the whole reason this is
        separate from ``Datablock.path``: that one names the canonical
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
# CheckpointPath — a checkpoint named by path, not by the run that wrote it
# ═══════════════════════════════════════════════════════════════════════

class CheckpointPath(CheckpointBuilder):
    """One checkpoint file, named literally, that another run can start from.

    ``Still`` answers ``find_latest_ckpt()`` by listing its own ``ckpts``
    topic, so warm-starting from an earlier run ordinarily means holding that
    run's block -- which means reconstructing its whole ``VAR``, and which
    stops being possible once the code that produced it has moved on (a
    ``VERSION`` bump, a field added, a module renamed).  This names the file
    instead, so a checkpoint can outlive the configuration that wrote it.

    The path is a ``VAR`` field, so it lands in the identity: two runs started
    from different checkpoints are different artifacts, and ``step=0319000``
    stays distinguishable from ``step=0318000`` of the same source.  That is
    the opposite trade from passing the source ``Still`` itself, whose key
    records *which* run supplied the weights and deliberately does **not**
    move as that run trains further.  Neither is the right answer in general:
    pick the one whose identity records what you will want to have recorded.
    Here it is the file, because a path is all that is left of a run whose
    still can no longer be built.

    A warm start, not a resume.  This is a ``CheckpointBuilder``, so ``Still``
    takes the weights only, at step 0, with a fresh optimizer.  Handing the
    same path to ``VAR.ckpt_builder`` as a bare **string** is the other
    behaviour -- a full resume, optimizer state and step counter included --
    which is what you want only when pointing a run at its own checkpoint that
    has moved.
    """

    VERSION = 1

    #: Writes nothing, and says so.  `TOPICS = []` renders into the identity as
    #: () where a class declaring no TOPICS at all renders as ("topics:None",);
    #: either would do for a new class with no artifacts to re-key, and this is
    #: the one that states the intent.
    TOPICS = []

    @dataclass
    class VAR(CheckpointBuilder.VAR):
        ckpt_path: str | None = None

    # 1. Datablock protocol ─────────────────────────────────────────

    def valid(self):
        """True when the checkpoint this block names is actually there.

        Not the base "a block with no topics is always valid": the single
        claim this block makes is that one file exists, so checking it turns a
        mistyped path into a failure now rather than a ``FileNotFoundError``
        however many hours into a run.

        ``Still`` does not consult this on its way to a build -- its
        ``ckpt_builder`` is in ``TREE_SKIP_VALIDATION`` -- so this answers a
        direct caller, and is worth calling before launching a run.
        """
        if self.var.ckpt_path is None:
            return False
        try:
            return self.fs.exists(str(self.var.ckpt_path))
        except Exception:
            return False

    def __build__(self):
        """Nothing to build: the checkpoint is another run's artifact.

        A no-op rather than a raise, unlike ``LightningBuilder.__build__``:
        reaching it is not evidence of a bug, since ``build_tree()`` may walk
        here perfectly legitimately.  This block only ever reads a file it did
        not write.
        """
        return self

    # 2. Declared API ───────────────────────────────────────────────

    def find_latest_ckpt(self, *, pull: bool = False) -> str | None:
        """The checkpoint this block names, or ``None`` when it names none.

        "Latest" is the name of the protocol method, not a claim being made
        about this file: there is exactly one here, and pinning it is the
        whole point.

        *pull* keeps ``Still.find_latest_ckpt``'s meaning.  False answers with
        the path as given -- possibly remote, not downloaded, not checked.
        True answers with a local and intact one, fetching it if need be,
        which is the form ``torch.load`` and Lightning's ``ckpt_path=`` both
        require.
        """
        if self.var.ckpt_path is None:
            return None
        src = str(self.var.ckpt_path)
        if not pull:
            return src

        if os.path.isfile(src):
            # Already a local file.  Returned as it stands: copying a multi-GB
            # checkpoint under our own key would spend the disk to gain
            # nothing, since we are not the ones who own or clean it up.
            self._require_intact_(src, src)
            return src

        if not self.fs.exists(src):
            raise FileNotFoundError(
                f"{type(self).__name__}: no checkpoint at {src} -- "
                f"VAR.ckpt_path names a file that is not there"
            )
        dest = self._local_ckpt_path_()
        if not is_intact_archive(dest):
            self.localfs.makedirs(os.path.dirname(dest), exist_ok=True)
            self.log.info("%s: fetching %s -> %s", type(self).__name__, src, dest)
            self.pull(src, dest, show_progress=True)
        self._require_intact_(dest, src)
        return dest

    # 3. Accessors ──────────────────────────────────────────────────

    @property
    def ckpt_step(self) -> int:
        """The training step the filename encodes, or -1 -- for a banner or a log."""
        if self.var.ckpt_path is None:
            return -1
        return Still._ckpt_step_(os.path.basename(str(self.var.ckpt_path)))

    # 4. Helpers ────────────────────────────────────────────────────

    def _local_ckpt_path_(self) -> str:
        """Where a remote checkpoint is staged: under this block's own local key.

        Its own key and not a shared scratch dir, so two `CheckpointPath`
        blocks naming same-named files from different runs -- which is the
        norm, since a step number is all that distinguishes them -- cannot
        land on top of each other.
        """
        return os.path.join(self.localanchorkeypath, os.path.basename(str(self.var.ckpt_path)))

    def _require_intact_(self, path, src):
        """Raise unless *path* is a readable archive, saying where it came from."""
        if not is_intact_archive(path):
            raise RuntimeError(
                f"{type(self).__name__}: {path} is not a readable checkpoint "
                f"archive (from {src}) -- it is truncated or was not a "
                f"checkpoint to begin with"
            )


# ═══════════════════════════════════════════════════════════════════════
# Still — one training run
# ═══════════════════════════════════════════════════════════════════════

class Still(CheckpointBuilder):
    """One Lightning training run, addressed by its configuration.

    Three topics: ``ckpts`` (the checkpoints) and ``logs`` (the TensorBoard run
    directories), both staged locally and synced to the block's storage as the
    run proceeds -- so a run whose machine dies leaves its checkpoints behind,
    and a resumed run finds them -- and ``done``, written once ``fit()``
    returns, which is the whole of what ``valid`` reads.

    What a subclass has to supply
    -----------------------------
    * The four required builder blocks in VAR -- ``model_builder``,
      ``lightning_builder``, ``training_dataset_builder`` and
      ``validation_dataset_builder``.  Construction fails without them, naming
      whichever is absent.  See the module docstring for why each is a block.
      Passing the *same* dataset builder for both draws validation as a
      held-out, block-granular split of the training data instead of building
      two datasets.
    * ``train_transform`` / ``val_transform`` / ``train_collate_fn`` /
      ``val_collate_fn`` on the LightningModule, if it wants any.  These are
      the model's business -- an augmentation pipeline and a collator are as
      model-specific as the forward pass -- so they live with the model rather
      than being configured here.  Each defaults to ``None``, i.e. plain
      tensors and the default collate.

    What it may override
    --------------------
    ``_banner_rows_`` (what the start-of-run banner reports),
    ``dataloaders``, ``trainer_kwargs``, ``callbacks``, and the
    ``Sampler`` / ``Loader`` / ``split_indices`` hooks.

    Operational parameters -- ``n_devices``, ``num_workers``,
    ``prefetch_factor``, ``tensorlogs_root``, ``save_remote_logs``,
    ``stall_timeout_s``, ``debug_share_train_val`` -- are constructor keywords,
    not VAR: they change how a run executes, not what it computes, so two runs
    differing only in those are the same run and share a key.
    """

    #: 2 since builders replaced the cfg_ surface: that re-keyed every
    #: still, and the bump says so in the key rather than leaving it
    #: implied by a changed signature.
    #: 3: ``done`` became a topic of its own, where completion used to be a
    #: ``_COMPLETE`` file inside ``ckpts``. TOPICS is in the signature, so
    #: every still re-keys; the bump is what puts that in the path rather than
    #: letting artifacts move silently. There is no read-compatibility with
    #: the old marker: a version=2 artifact is reached by redirecting its
    #: ``ckpts``/``logs`` and then calling ``UNSAFE_done()``, which writes
    #: the new topic.
    VERSION = 3
    #: ``done`` is a topic and not a marker file inside ``ckpts`` because
    #: completion is a thing this block produces, and dbx already knows how to
    #: write, validate, clear, copy and redirect one of those.  As a file it
    #: needed five hand-rolled special cases -- a local-then-remote existence
    #: check in ``valid``, a two-location write in ``UNSAFE_done``, a name
    #: excluded from every checkpoint listing, and a line in the class
    #: docstring explaining that ``ckpts`` holds something that is not a
    #: checkpoint.  Matches ``DatapointTable``, which has always done it this
    #: way.
    #:
    #: The dict form, because the three are not alike: ``ckpts`` and ``logs``
    #: are directories (:class:`DIR`) and ``done`` is one file.
    TOPICS = {'ckpts': DIR, 'logs': DIR, 'done': 'done'}

    # A warm-start source in `ckpt_builder` needs *a* checkpoint, not a finished run:
    # branching off a still that is still training, or that stopped before
    # writing its done topic, is a legitimate and common thing to do.
    # TREE_SKIP_BUILDING keeps build_tree() from training the source on your
    # behalf; TREE_SKIP_VALIDATION keeps __pre_build__'s validate_vars pass from
    # rejecting the whole build with a generic "not all upstream Datablocks in
    # var are valid" -- where __build__ would otherwise say precisely which
    # still has no checkpoints.
    TREE_SKIP_BUILDING = ('ckpt_builder',)
    TREE_SKIP_VALIDATION = ('ckpt_builder',)
    #: Sampler and loader for ``dataloaders``.  Both default to the
    #: shard-locality-aware pair in ``dbx.datastreams``; a subclass whose
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
        # --- What to train with: every collaborator is a block ---
        # The first four are REQUIRED; __check_builders__ refuses to construct
        # without them. They carry a None default only because dbx builds
        # VAR() with no arguments for a block that has no spec at all
        # (Datablock.__setstate__), where a dataclass-required field would
        # fail about positional arguments instead of about what is missing.
        model_builder: ModelBuilder | str | None = None
        lightning_builder: LightningBuilder | str | None = None
        training_dataset_builder: DatasetBuilder | str | None = None
        validation_dataset_builder: DatasetBuilder | str | None = None
        # --- Where to start ---
        # None: resume this run's own latest checkpoint, if it has one.
        # A CheckpointBuilder: warm-start from *its* latest -- weights only, at
        #   step 0, fresh optimizer, since another run's step count and moments
        #   say nothing about this one. Consulted only until this run has a
        #   checkpoint of its own; after that a resume continues this run.
        # A path: resume it in full, optimizer and step count included, which
        #   is what you want when pointing a run at its own checkpoint that
        #   has moved.
        # See _resolve_resume_ckpt_().
        ckpt_builder: CheckpointBuilder | str | None = None
        # --- How to split and shuffle ---
        train_val_split: float = 0.8
        dataset_seed: int = 42
        block_shuffle_size: int = 2048   # samples per shuffle block; see Sampler
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
        max_training_steps: int | None = None
        # Every *_n_steps field below counts OPTIMIZER steps, not micro-batches
        # -- see the val_check_interval scaling in trainer_kwargs(). 1 = every step.
        train_log_every_n_steps: int = 10
        val_every_n_steps: int = 500
        val_max_batches: int | None = 1
        val_shuffle: bool = False
        val_seed: int = 42
        gradient_clip_val: float = 1.0
        gradient_clip_algorithm: str = 'norm'
        ckpt_every_n_steps: int | None = None
        ckpt_every_n_epochs: int = 5
        precision: str | None = 'bf16-mixed'
        matmul_precision: str | None = 'high'
        accumulate_grad_batches: int = 2
        batch_size: int = 16
        still_seed: int = 42

    # 1. Datablock protocol ─────────────────────────────────────────

    def __init__(
        self,
        *,
        n_devices=1,
        devices=None,
        strategy=None,
        check_run=False,
        tensorlogs_root=None,
        # Whether a weights-only load must match the model exactly. True by
        # default: "these weights fit this architecture" is the claim a warm
        # start makes, and it should have to hold. Pass False for the case
        # where it deliberately does not -- a warm start into a changed head,
        # which has missing keys by construction. See _load_weights_only_.
        strict_loading=True,
        save_remote_logs=True,
        debug_share_train_val=False,
        num_workers=4,
        prefetch_factor=4,   # PyTorch's default is 2; bumped to smooth I/O-bound stalls
        # Seconds one dataset item may take before its dataloader worker is
        # declared wedged. 0 or None disables it. Only does anything if the
        # subclass returns a watchdog from dataloader_worker_init_fn().
        stall_timeout_s=None,
        **kwargs,
    ):
        # Threaded through super() rather than assigned directly: the base
        # records **kwargs in self.parameters, which is what __getstate__
        # serializes, so .set(tag=...) preserves them. Plain `self.x = ...`
        # would silently vanish on rebuild -- and every pipeline entrypoint
        # calls .set(tag=...).
        if devices is not None and n_devices != 1:
            raise ValueError(
                f"pass devices or n_devices, not both -- they name the same "
                f"thing (got devices={devices!r}, n_devices={n_devices!r})"
            )
        super().__init__(
            n_devices=n_devices,
            devices=devices,
            strategy=strategy,
            check_run=check_run,
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
        self.__check_builders__()

    #: VAR fields naming a block this still cannot run without.  ``ckpt_builder``
    #: is deliberately absent: a run with nothing to warm-start from is the
    #: ordinary case.
    REQUIRED_BUILDERS = ('model_builder', 'lightning_builder',
                         'training_dataset_builder', 'validation_dataset_builder')

    def __check_builders__(self):
        """Refuse to construct without every builder this still trains from.

        A still delegates all of its construction, so a missing builder is not
        a value that defaults sensibly -- it is a run that cannot be described,
        let alone executed.  Caught here, on construction, rather than hours
        later inside ``__build__``.

        The fields carry a ``None`` default despite being required because dbx
        builds ``VAR()`` with no arguments for a block that has no spec at all
        (``Datablock.__setstate__``); a dataclass-required field would raise
        there, about positional arguments, instead of here, about what is
        actually missing.
        """
        missing = [n for n in self.REQUIRED_BUILDERS if getattr(self.var, n) is None]
        if missing:
            raise TypeError(
                f"{type(self).__name__}: VAR {missing} is None, and a still builds "
                f"nothing itself -- every collaborator is a block it asks. Pass one "
                f"per field in spec=, e.g. spec=dict(model_builder=MyModel(...), "
                f"lightning_builder=MyLightning(...)). See dbx.stills"
            )

    def __build__(self):
        """Run the training loop."""
        logs_dir = self._local_logs_dir_
        ckpts_dir = self._local_ckpts_dir_
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(ckpts_dir, exist_ok=True)

        # Registered before anything can fail: a run killed by SIGINT, an OOM,
        # or a preempted node has checkpoints on local disk that are worth more
        # than the exception, and nothing else would ever move them.
        def _atexit_sync():
            self.log.info("atexit: syncing checkpoints to remote...")
            self._sync_to_remote_(reason='atexit')

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

            resume_plan = self._resume_plan_()
            self._print_training_banner_(model, resume_plan, ckpts_dir)

            ckpt, own_resume_ckpt, warm_start = self._resolve_resume_ckpt_()
            self._report_resume_divergence_(ckpt, resume_plan, warm_start=warm_start)

            fit_kwargs = {}
            if ckpt is not None:
                if warm_start or self.var.reset_optimizer_state:
                    self._load_weights_only_(
                        model, ckpt,
                        why=("warm start from another block" if warm_start
                             else "optimizer reset"),
                    )
                    # Fully read by the load above -- free it now rather than
                    # leaving it on disk for the run's duration.
                    if own_resume_ckpt:
                        self._free_local_ckpt_(ckpt)
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
                                _still._free_local_ckpt_(_resume_ckpt_path)

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

            if self.check_run:
                # A check is not a trained model. It fits a batch or two and
                # returns, and marking the block complete here would leave a
                # key that claims a finished run and a `valid()` that agrees --
                # so the next real build would skip it entirely.
                self.log.warning(
                    "check_run=%r: ran %s batch(es), NOT writing the done "
                    "topic. This block is still unbuilt.",
                    self.check_run,
                    self.check_run if self.check_run is not True else 1,
                )
            else:
                self.UNSAFE_done(OVERRIDE=True)
            self._sync_to_remote_(reason='post-fit')
        finally:
            # Unregistered first: the sync below does the same work, and an
            # atexit handler that runs afterwards would repeat a multi-GB push.
            atexit.unregister(_atexit_sync)
            self._sync_to_remote_(reason='finally')
            torch.set_float32_matmul_precision(original_matmul_precision)

        return self

    def valid(self):
        """True only when training has run to completion.

        The ``done`` topic, not "are there checkpoints": a run that stopped at
        epoch 3 of 30 has checkpoints and is not done, and treating it as valid
        would make ``build_tree()`` hand a half-trained encoder to everything
        downstream.

        Local staging first, then the topic proper, so a run that finished on
        this machine answers without a network round trip.  Both are the
        topic's own paths -- ``path('done', local=True)`` and
        ``valid_topic('done')`` -- rather than a filename this method knows and
        nothing else does.
        """
        local_done = self.path('done', local=True)
        if local_done is not None and os.path.exists(local_done):
            return True
        try:
            return self.valid_topic('done')
        except Exception:
            return False

    # 2. Declared API ───────────────────────────────────────────────

    def linklogs(self):
        """Symlink this run's local log dir under ``tensorlogs_root``.

        So that one ``tensorboard --logdir`` sees every run, named by key,
        without the logs themselves moving out of the block's staging area.
        """
        return self.linklocal('logs', self._logslink_)

    # ── UNSAFE_ helpers ────────────────────────────────────────────

    def UNSAFE_done(self, *, OVERRIDE: bool = False):
        """Write the ``done`` topic locally and remotely, forcing ``valid``.

        ``__build__`` calls this once ``trainer.fit()`` returns.  It is
        also directly callable to hand-declare a run complete -- which is how
        a version=2 still is migrated: redirect its ``ckpts`` and ``logs`` to
        where they already are, then call this to write the topic that
        replaced its ``_COMPLETE`` marker.  Also after assembling checkpoints
        with ``UNSAFE_copy_from`` from a source that carried no ``done`` of
        its own.

        Each location is written directly rather than by pushing the topic,
        because this has to be safe to call when local staging holds no
        checkpoints at all (right after a copy that went straight to remote),
        where a local-to-remote directory push would overwrite real remote
        checkpoints with an empty local directory.  ``valid`` checks local
        then remote, so either alone would do; writing both keeps staging and
        storage consistent with each other, matching what a real run leaves
        behind.
        """
        if not UNSAFE_allowed("UNSAFE_done", OVERRIDE=OVERRIDE):
            return self
        timestamp = datetime.now().isoformat() + "\n"
        local_done = self.path('done', local=True, ensure_dirpath=True)
        with open(local_done, "w") as f:
            f.write(timestamp)
        if not self.is_local_fs:
            with self.fs.open(self.path('done', ensure_dirpath=True), "w") as f:
                f.write(timestamp)
        self.log.info(
            "UNSAFE_done: wrote the done topic (%s)",
            "local" if self.is_local_fs else "local + remote",
        )
        return self

    def UNSAFE_clear(self, *topics, OVERRIDE: bool = False, clear_dirpath: bool = False):
        """As the base, plus ``done``, the local staging dirs and the TB symlink.

        **Clearing ``ckpts`` clears ``done``.**  They are separate topics but
        not independent ones: ``done`` asserts that the checkpoints it was
        written beside are a finished run, so throwing those away and leaving
        it behind would leave ``valid`` reporting a trained model with no
        weights to show for it -- and ``build_tree`` skipping the block that
        would rebuild them.  This is the one coupling the topic split costs,
        and it buys the five special cases the ``_COMPLETE`` file needed.

        Not the reverse: clearing ``done`` alone is how you say "these
        checkpoints are real but the run is not finished", which is what
        reopening a run that was marked complete too early requires.

        Without the rest of this, clearing a topic removes its remote copy and
        leaves local staging behind -- so the next ``valid`` answers from a
        stale local file and the block reports itself built.
        """
        if topics and 'ckpts' in topics and 'done' not in topics:
            topics = (*topics, 'done')
        result = super().UNSAFE_clear(*topics, OVERRIDE=OVERRIDE, clear_dirpath=clear_dirpath)
        if len(topics) == 0 or 'logs' in topics:
            local_logs = self._local_logs_dir_
            if os.path.isdir(local_logs):
                shutil.rmtree(local_logs, ignore_errors=True)
            logslink = self._logslink_
            if logslink and os.path.lexists(logslink):
                try:
                    os.remove(logslink)
                except OSError:
                    pass
        if len(topics) == 0 or 'ckpts' in topics:
            local_ckpts = self._local_ckpts_dir_
            if os.path.isdir(local_ckpts):
                shutil.rmtree(local_ckpts, ignore_errors=True)
        return result

    def UNSAFE_clear_cache(self, *, OVERRIDE: bool = False):
        """Remove the local staging caches without touching remote storage.

        Deletes ``_local_workdir_`` (the ``ckpts/`` and ``logs/`` subdirs
        under dbx's ``local=`` staging root) and the TensorBoard symlink.  The
        remote topics are left intact, so the run's record survives -- call
        this on a machine that has finished training and needs its disk back.
        """
        if not UNSAFE_allowed("UNSAFE_clear_cache", OVERRIDE=OVERRIDE):
            return self
        workdir = self._local_workdir_
        if os.path.isdir(workdir):
            self.log.info("UNSAFE_clear_cache: removing local workdir %s", workdir)
            shutil.rmtree(workdir, ignore_errors=True)
        logslink = self._logslink_
        if logslink and os.path.lexists(logslink):
            self.log.info("UNSAFE_clear_cache: removing logslink %s", logslink)
            try:
                os.remove(logslink)
            except OSError as e:
                self.log.warning("UNSAFE_clear_cache: could not remove logslink: %s", e)
        return self

    def UNSAFE_copy_from(self, anchorkeypath, *, ckpts: int = 0, **kwargs):
        """As ``~dbx.Datablock.UNSAFE_copy_from``, but ``ckpts`` may be a subset.

        Parameters
        ----------
        ckpts : int, default 0
            0 copies every file under the source ``ckpts`` topic, as the base
            does.  A positive N copies only the earliest N by training step; a
            negative N only the most recent N.

            ``done`` is a topic of its own and so is copied by the base along
            with everything else, which is what keeps post-copy ``valid`` (and
            the default ``validate=True``) passing.  A source that has none --
            an unfinished run -- copies without one, and
            ``UNSAFE_done()`` is how you declare the result complete.
        **kwargs
            Forwarded to the base (``OVERRIDE``, ``overwrite``, ``topicpaths``,
            ``validate``, ``always_copy_whole_dirpath``, ``show_progress``).
        """
        return super().UNSAFE_copy_from(anchorkeypath, ckpts=ckpts, **kwargs)

    # ── Checkpoints ────────────────────────────────────────────────

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
            return self._find_latest_ckpt_remote_()

        os.makedirs(self.dirpath('ckpts', local=True), exist_ok=True)

        def _valid_ckpt(path):
            return self._is_intact_ckpt_(path, log=self.log)

        try:
            result = self.synclocal(
                'ckpts', suffix='.ckpt', key=self._ckpt_step_, validate=_valid_ckpt,
                latest=True, show_progress=True,
            )
        except Exception as e:
            self.log.verbose(
                f"find_latest_ckpt: remote check failed, falling back to local scan: {e}")
            result = None
            ckpts_dir = self.dirpath('ckpts', local=True)
            candidates = sorted(
                (f for f in os.listdir(ckpts_dir) if f.endswith(".ckpt")),
                key=self._ckpt_step_,
            )
            for name in reversed(candidates):
                path = os.path.join(ckpts_dir, name)
                if _valid_ckpt(path):
                    result = path
                    break

        if result:
            self.log.info(
                "find_latest_ckpt: using %s (step=%s)",
                os.path.basename(result), self._ckpt_step_(os.path.basename(result)),
            )
        return result

    # ── Dataloaders ────────────────────────────────────────────────

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

    #: Every hook ``dataloaders`` asks the module for. All optional. The
    #: canonical list, so that ``module_hooks`` can report what a module
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

    def dataloader_worker_init_fn(self):
        """The ``worker_init_fn`` every loader this still builds is given.

        The worker is a torch DATALOADER worker -- one of the ``num_workers``
        subprocesses feeding batches in -- not a dbx parallelization worker,
        which is a different thing entirely and configured elsewhere.

        ``None`` here, so plain torch behaviour.  Override to install a
        per-worker guard -- detaching inherited stdio, arming a stall watchdog.
        Whatever it returns must be picklable, since a ``spawn``-context
        DataLoader has to send it to the worker.
        """
        return None

    def dataloaders(self, model=None, debug_share_train_val=None):
        """The ``(train, val)`` DataLoaders, exactly as ``__build__`` uses them.

        Runs no training -- useful for inspecting the dataset/collation wiring
        directly, e.g. to confirm the images reaching the model are meaningful.

        Parameters
        ----------
        model : LightningModule, optional
            Reused when already built (as during ``__build__``); otherwise
            taken from ``lightning_module``, which caches, so repeated
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
        self.log.debug("module hooks: %s", self.module_hooks(model))
        train_transform = self._module_hook_(model, 'train_transform')
        val_transform = self._module_hook_(model, 'val_transform')
        train_kwargs = self._module_hook_(model, 'train_dataset_kwargs', {}) or {}
        val_kwargs = self._module_hook_(model, 'val_dataset_kwargs', {}) or {}
        block_size = var.block_shuffle_size

        # Same block for both fields means "hold validation out of the training
        # data" -- one dataset, two disjoint index subsets. Two different blocks
        # mean two independently built datasets. Said by what is in VAR rather
        # than by leaving a field empty, now that both fields are required: the
        # split is a choice the spec states, not an omission it implies.
        if val_builder is not train_builder:
            train_dataset = train_builder.dataset(transform=train_transform, **train_kwargs)
            val_dataset = val_builder.dataset(transform=val_transform, **val_kwargs)
            train_collate_fn = self._module_hook_(model, 'train_collate_fn')
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
            train_collate_fn = self._module_hook_(
                model, 'shared_train_collate_fn',
                self._module_hook_(model, 'train_collate_fn'),
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

        training_dataloader = self._dataloader_(
            train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            collate_fn=train_collate_fn,
            sampler=train_sampler,
            loader_cls=self.Loader,
        )
        val_dataloader = self._dataloader_(
            val_dataset,
            batch_size=self.val_batch_size,
            num_workers=val_workers,
            prefetch_factor=val_prefetch,
            collate_fn=self._module_hook_(model, 'val_collate_fn'),
            sampler=val_sampler,
        )
        return training_dataloader, val_dataloader

    # ── Trainer assembly ───────────────────────────────────────────

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

        accelerator, devices = self._resolve_devices_()
        kwargs['devices'] = devices
        if accelerator is not None:
            kwargs['accelerator'] = accelerator
        if self.strategy is not None:
            kwargs['strategy'] = self.strategy
        if self.check_run:
            # Lightning's own name for it. True is one batch and an int is
            # that many, of train AND val. It silences the loggers and any
            # ModelCheckpoint -- but NOT this still's own upload callbacks,
            # and not __build__'s UNSAFE_done, which is why __build__
            # skips `done` itself rather than trusting this flag to.
            kwargs['fast_dev_run'] = self.check_run
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

    # 3. Accessors and properties ───────────────────────────────────

    @property
    def lightning_module(self):
        """The ``LightningModule`` to train, built once and kept.

        Asked of ``VAR.lightning_builder``, which is the block that knows how
        to build it -- and whose identity this still's key already depends on.
        """
        if getattr(self, '_lightning_module', None) is None:
            self._lightning_module = self.var.lightning_builder.lightning_module
        return self._lightning_module

    def model(self):
        """The model to train, from ``VAR.model_builder``.

        A fresh instance per call, the same as asking the builder directly.
        """
        return self.var.model_builder.model()

    @property
    def train_batch_size(self) -> int:
        """Batch size for the training loader."""
        return self.var.batch_size

    @property
    def val_batch_size(self) -> int:
        """Batch size for the validation loader.

        Separate from ``train_batch_size`` so a subclass whose validation
        metric depends on its own batch size -- an in-batch retrieval score,
        where chance is ``1/B`` -- can vary one without the other.
        """
        return self.var.batch_size

    # 4. Private methods and helpers ────────────────────────────────

    # ── Local working directories ──────────────────────────────────

    #
    # Backed by dbx's local= staging (Datablock.localanchorkeypath /
    # .dirpath(topic, local=True)) rather than hand-parsing anchorkeypath.
    # When the primary url is itself local storage, dbx aliases localfs/localroot
    # to fs/root, so these transparently collapse to the non-local
    # .path()/.dirpath() equivalents -- no branching needed here for that case.

    @property
    def _local_workdir_(self):
        return self.localanchorkeypath

    @property
    def _local_logs_dir_(self):
        return self.dirpath('logs', local=True)

    @property
    def _local_ckpts_dir_(self):
        return self.dirpath('ckpts', local=True)

    @property
    def _logslink_(self):
        if self.tensorlogs_root is None:
            return None
        return os.path.join(self.tensorlogs_root, self.key)

    # ── UNSAFE_ helpers ────────────────────────────────────────────

    def _UNSAFE_copy_topic_(self, topic, anchorkeypath, *, ckpts: int = 0, **kwargs):
        if topic != 'ckpts' or not ckpts or kwargs.get('always_copy_whole_dirpath'):
            return super()._UNSAFE_copy_topic_(topic, anchorkeypath, **kwargs)
        self._UNSAFE_copy_ckpts_subset_(anchorkeypath, count=ckpts,
                                        topicpaths=kwargs.get('topicpaths'))

    def _UNSAFE_copy_ckpts_subset_(self, anchorkeypath, *, count, topicpaths=None):
        """Copy the earliest (``count > 0``) or latest (``count < 0``) checkpoints."""
        _src_path = topicpaths['ckpts'] if topicpaths is not None else 'ckpts'
        src_path = os.path.join(anchorkeypath, _src_path)
        src_fs, _ = self._url_to_fs(src_path)
        if not src_fs.exists(src_path):
            return
        entries = [os.path.basename(p.rstrip('/')) for p in src_fs.ls(src_path, detail=False)]
        # Only .ckpt files: `done` is a sibling topic now, copied by the base.
        ckpt_names = sorted((n for n in entries if n.endswith('.ckpt')), key=self._ckpt_step_)
        selected = ckpt_names[:count] if count > 0 else ckpt_names[count:]
        self.log.verbose(
            f"UNSAFE_copy_from: ckpts={count} -> copying {len(selected)}/{len(ckpt_names)} "
            f"checkpoint(s): {selected}"
        )
        dst_dir = self.dirpath('ckpts', ensure=True)
        for name in selected:
            # _UNSAFE_copy_file prefers a direct server-side blob copy when src
            # and dst are on the same remote filesystem (the common case here:
            # two stills' ckpts/ under one storage account) -- which matters
            # because checkpoints are large, and the generic get-then-put
            # fallback round-trips every byte through local disk.
            self._UNSAFE_copy_file(
                os.path.join(src_path, name),
                os.path.join(dst_dir, name),
            )

    # ── Checkpoints ────────────────────────────────────────────────

    #: Matches both ``step=1234`` and the doubled ``step=step=1234`` that
    #: Lightning's own ModelCheckpoint used to emit.
    CKPT_STEP_RE = re.compile(r'step=(?:step=)?(\d+)')

    @classmethod
    def _ckpt_step_(cls, name):
        """The training step a checkpoint filename encodes, or -1."""
        m = cls.CKPT_STEP_RE.search(name)
        return int(m.group(1)) if m else -1

    @staticmethod
    def _is_intact_ckpt_(path, log=None):
        """``is_intact_archive``, saying which checkpoint it rejected and why."""
        ok = is_intact_archive(path)
        if not ok and log is not None:
            log.verbose(f"Skipping corrupt checkpoint {os.path.basename(str(path))}")
        return ok

    def _find_latest_ckpt_remote_(self):
        """The remote path of the latest ``.ckpt``, without pulling or validating."""
        try:
            entries = [(os.path.basename(e.rstrip('/')), e) for e in self.ls('ckpts')]
        except Exception as e:
            self.log.verbose(f"find_latest_ckpt(pull=False): remote listing failed: {e}")
            return None
        candidates = sorted(
            ((name, path) for name, path in entries if name.endswith('.ckpt')),
            key=lambda item: self._ckpt_step_(item[0]),
        )
        if not candidates:
            return None
        latest_name, latest_path = candidates[-1]
        self.log.info(
            "find_latest_ckpt(pull=False): latest is %s (step=%s) at %s -- not downloaded",
            latest_name, self._ckpt_step_(latest_name), latest_path,
        )
        return latest_path

    def _free_local_ckpt_(self, local_path):
        """Delete a local checkpoint that is known to be duplicated on remote.

        Used for the checkpoint a run resumes *from*: ``find_latest_ckpt``
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

    # ── Sync helpers ───────────────────────────────────────────────

    #
    # pushtopic(topic) pushes the whole local-staged topic dir to its canonical
    # remote path, no-oping on its own when they are the same path (local
    # storage) -- dbx's push() checks src == dest directly. The is_local_fs
    # guards below only skip the (otherwise misleading) "syncing to remote" log
    # line when there is no separate remote to sync to.

    def _sync_ckpts_to_remote_(self):
        if self.is_local_fs:
            return
        self.log.info("Syncing ckpts to remote")
        self.pushtopic('ckpts')

    def _sync_logs_to_remote_(self):
        if self.is_local_fs:
            return
        self.log.info("Syncing logs to remote")
        self.pushtopic('logs')

    def _sync_to_remote_(self, *, reason):
        """Push both topics, never raising -- called from ``atexit`` and ``finally``."""
        try:
            self._sync_ckpts_to_remote_()
        except Exception as e:
            self.log.warning("%s: ckpt sync failed: %s", reason, e)
        if self.save_remote_logs:
            try:
                self._sync_logs_to_remote_()
            except Exception as e:
                self.log.warning("%s: log sync failed: %s", reason, e)

    # ── Dataloaders ────────────────────────────────────────────────

    @staticmethod
    def _module_hook_(module, name, default=None):
        """Call *module*'s *name* hook if it has one, else return *default*."""
        hook = getattr(module, name, None)
        if hook is None:
            return default
        return hook() if callable(hook) else hook

    def _dataloader_(self, dataset, *, batch_size, num_workers, prefetch_factor,
                     collate_fn, sampler, loader_cls=None):
        """Build one DataLoader.  The single place a loader is constructed.

        Both loaders come through here so that ``dataloader_worker_init_fn``
        cannot be passed to one and forgotten on the other -- which is exactly what
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
            worker_init_fn=self.dataloader_worker_init_fn(),
            **kwargs,
        )

    # ── Training banner ────────────────────────────────────────────

    #
    # Split out of __build__ so a subclass with a different architecture can
    # restate the rows without duplicating the training loop around them.

    @staticmethod
    def _banner_trunc_(s, w):
        s = str(s)
        return s if len(s) <= w else s[:w - 3] + "..."

    @staticmethod
    def _fmt_params_(n):
        if n >= 1e9: return f"{n / 1e9:.1f}B"
        if n >= 1e6: return f"{n / 1e6:.1f}M"
        if n >= 1e3: return f"{n / 1e3:.1f}K"
        return str(n)

    def _banner_param_str_(self, model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return f"{self._fmt_params_(total)} (trainable: {self._fmt_params_(trainable)})"

    def _banner_dataset_tag_(self):
        return getattr(self.var.training_dataset_builder, 'tag', None) \
            or type(self.var.training_dataset_builder).__name__

    def _banner_rows_(self, model, ckpt, ckpts_dir):
        """``[(label, value), ...]`` for the banner.  Extend in a subclass."""
        var = self.var
        _W = self.BANNER_WIDTH
        return [
            ("PARAMS",    self._banner_param_str_(model)),
            ("DATASET",   str(self._banner_dataset_tag_())),
            ("EPOCHS",    str(var.max_epochs)),
            ("BATCH",     f"{var.batch_size} x {var.accumulate_grad_batches} accum"
                          f" = {var.batch_size * var.accumulate_grad_batches} eff."),
            ("PRECISION", str(var.precision)),
            # RESUME, not CKPT: this is the optimizer/step state the run
            # continues from, which is a different question from where the
            # weights came from. "CKPT None" read as "no weights loaded".
            ("RESUME",    self._banner_trunc_(
                ckpt if ckpt else "none - fresh run at step 0", _W - 14)),
            ("LOCALWORK", self._banner_trunc_(ckpts_dir, _W - 14)),
            ("KEY",       self._banner_trunc_(self.key, _W - 14)),
        ]

    def _print_training_banner_(self, model, ckpt, ckpts_dir):
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
        for _label, _val in self._banner_rows_(model, ckpt, ckpts_dir):
            _line = f"  {_label:<10}{_val}"
            print("║" + self._banner_trunc_(_line, _W).ljust(_W) + "║")
        print(_footer)

    # ── Resume ─────────────────────────────────────────────────────

    def _resume_plan_(self):
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
            plan = self._find_latest_ckpt_remote_()
        if plan is None and self.var.ckpt_builder is not None:
            plan = (self.var.ckpt_builder.key if hasattr(self.var.ckpt_builder, 'key')
                    else str(self.var.ckpt_builder))
            # Says which of the two things the banner's RESUME row means. They
            # are not the same run: continuing this one keeps its optimizer and
            # step count, starting from another one keeps only its weights.
            if self._warm_start_block_() is not None:
                plan = f"warm start, weights only - {plan}"
        return plan

    def _warm_start_block_(self):
        """``VAR.ckpt_builder`` when it is another *block* to start from, else None.

        A block is asked for its own latest checkpoint; a bare path is taken as
        given.  The distinction decides how the checkpoint is loaded -- see
        ``_resolve_resume_ckpt_``.
        """
        initial = self.var.ckpt_builder
        return initial if isinstance(initial, CheckpointBuilder) else None

    def _resolve_resume_ckpt_(self):
        """``(local_ckpt_or_None, own_it, warm_start)`` -- what to start from.

        ``own_it`` is True only for our *own* latest checkpoint: that is always
        a local path under the staging ``ckpts`` dir which is also safely
        duplicated on remote (that is how it got there -- downloaded from
        remote, or already uploaded by the save callback), so it is safe to
        free once loaded.  Never True for a ``var.ckpt_builder``-derived path: that may
        point at another block's cache or an arbitrary user-supplied file,
        neither of which we own or know to be remote-backed.

        ``warm_start`` is True when the checkpoint came from *another block*
        (``_warm_start_block_``), which decides how it is loaded: weights
        only, at step 0, with a fresh optimizer.  Another run's optimizer
        moments and step counter say nothing about this one, and carrying them
        over is worse than useless -- a source that already reached its own
        ``max_epochs`` leaves this run with nothing left to do, so it "trains"
        instantly and marks ``done`` over an untrained model.

        A bare *path* in ``var.ckpt_builder`` keeps the full-resume behaviour: it is
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
        if ckpt is None and self.var.ckpt_builder is not None:
            initial = self._warm_start_block_()
            if initial is not None:
                ckpt = initial.find_latest_ckpt(pull=True)
                warm_start = True
                if ckpt is None:
                    raise FileNotFoundError(
                        f"the warm-start block in VAR.ckpt_builder has no checkpoints: {initial}"
                    )
            else:
                ckpt = str(self.var.ckpt_builder)
        return ckpt, own_it, warm_start

    def _resolve_devices_(self):
        """``(accelerator, devices)`` for the Trainer, from ``devices`` or ``n_devices``.

        ``n_devices`` is the old surface and still works: it names a *count*
        and leaves the accelerator to Lightning's ``"auto"``.  It cannot say
        *which* device, and cannot ask for CPU at all on a box that has a GPU.

        ``devices`` names them::

            ['cuda']                  -> accelerator='cuda', devices=1
            ['cuda:1', 'cuda:2']      -> accelerator='cuda', devices=[1, 2]
            ['cuda', 'cuda']          -> accelerator='cuda', devices=2
            ['cpu']                   -> accelerator='cpu',  devices=1
            ['cpu', 'cpu']            -> accelerator='cpu',  devices=2

        The last is the one worth having: two CPU processes is a real DDP
        world, so the distributed paths -- the sampler splitting, the rank-0
        guards, the collective in the loss -- can be exercised on a laptop and
        in CI, without a GPU and without taking one away from a training run
        to do it.  Lightning picks DDP off ``devices > 1`` by itself; pass
        ``strategy=`` for a specific one.

        A bare int, ``-1`` or ``'auto'`` is handed through untouched, so
        anything Lightning's own ``devices`` accepts still works.
        """
        spec = self.devices
        if spec is None:
            return None, self.n_devices
        if isinstance(spec, int) or spec in ('auto', -1):
            return None, spec
        if isinstance(spec, str):
            spec = [spec]
        if not isinstance(spec, (list, tuple)) or not spec:
            raise ValueError(
                f"devices must be a non-empty list of device strings, an int, "
                f"-1 or 'auto' -- got {self.devices!r}"
            )

        kinds, indices = [], []
        for entry in spec:
            if not isinstance(entry, str):
                raise ValueError(
                    f"devices entries are strings like 'cuda', 'cuda:1' or "
                    f"'cpu' -- got {entry!r} in {self.devices!r}"
                )
            kind, _, index = entry.partition(':')
            kinds.append(kind.strip())
            indices.append(index.strip() or None)

        if len(set(kinds)) > 1:
            raise ValueError(
                f"devices must all name one accelerator -- got {sorted(set(kinds))} "
                f"in {self.devices!r}. One Trainer runs on one accelerator."
            )
        kind = kinds[0]
        named = [i for i in indices if i is not None]
        if named and len(named) != len(indices):
            raise ValueError(
                f"devices must be all indexed or all bare -- got {self.devices!r}. "
                f"Indexed picks those devices; bare repeats name a count."
            )
        if not named:
            return kind, len(spec)

        if kind == 'cpu':
            # CPUAccelerator counts processes; there is no cpu:1 to select.
            raise ValueError(
                f"cpu devices have no index -- got {self.devices!r}. "
                f"Repeat 'cpu' to ask for that many processes: ['cpu', 'cpu']."
            )
        try:
            wanted = [int(i) for i in named]
        except ValueError:
            raise ValueError(
                f"device indices must be integers -- got {self.devices!r}"
            ) from None
        if len(set(wanted)) != len(wanted):
            raise ValueError(f"devices names a device twice: {self.devices!r}")
        return kind, wanted

    def _load_weights_only_(self, model, ckpt, *, why):
        """Load *ckpt*'s weights into *model*, leaving the run at step 0.

        No optimizer state, no epoch or step counter: the caller wants the
        weights and a fresh training schedule.  *why* names the reason in the
        log line, since the two callers are quite different situations.

        ``strict_loading`` decides how much of a mismatch is tolerable, and
        the default is ``True``: "these weights fit this architecture" is what
        a warm start asserts, and the assertion should have to hold.  A
        renamed parameter, a changed width or a head that grew is then an
        error here, rather than a quietly randomly-initialised tensor that
        surfaces much later as a loss curve starting too high.  It is also the
        cheapest way to ask whether a checkpoint fits at all, before spending
        an epoch finding out.

        Pass ``strict_loading=False`` for the case where the mismatch is the
        point -- a warm start into a changed head, which has missing keys by
        construction.  Then the load is partial and the counts below say how
        partial.

        Either way, a checkpoint whose parameter *names* have nothing to do
        with this model is refused: under ``False`` that would load nothing at
        all, silently, and the run would train from random init while its
        banner said otherwise.

        ``map_location='cpu'``, so the checkpoint does not get to choose the
        device.  A Trainer that fitted on ``cuda:0`` saved CUDA tensors, and
        ``torch.load`` would otherwise honour that: it raises outright on a
        CPU-only box ("Attempting to deserialize object on a CUDA device"), and
        on a machine with fewer GPUs than the one that saved it raises an
        invalid-device error -- so a warm-start source becomes unusable
        anywhere but the hardware it was trained on.  Even where it succeeds it
        allocates a second, GPU-resident copy of the whole state dict, held
        until this function returns; for a warm start off a large encoder that
        is gigabytes of accelerator memory competing with the model that is
        about to train.  ``load_state_dict`` copies into the model's own
        parameters wherever the source tensors live, so staging on the host
        costs nothing and is what makes the checkpoint portable.
        """
        self.log.info("Loading weights only (%s) from %s", why, ckpt)
        checkpoint = torch.load(ckpt, map_location='cpu', weights_only=False)
        if "state_dict" not in checkpoint:
            raise KeyError(
                f"{ckpt} has no 'state_dict' to warm-start from "
                f"(top-level keys: {sorted(checkpoint)})"
            )
        state_dict = checkpoint["state_dict"]
        if self.strict_loading:
            # load_state_dict(strict=True) raises with both lists in the
            # message, which is the report wanted here: what this model has
            # that the checkpoint does not, and the reverse.
            missing, unexpected = model.load_state_dict(state_dict, strict=True)
        else:
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

    def _report_resume_divergence_(self, ckpt, resume_plan, *, warm_start=False):
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


# ═══════════════════════════════════════════════════════════════════════
#  The names these classes used to have
# ═══════════════════════════════════════════════════════════════════════

#: This file was ``dbx/datastills.py``. It gets no module alias, unlike
#: ``dbx.backbones`` and ``dbx.probes``: it is a week old, has never been
#: released, and no artifact anywhere is stored under a ``dbx.datastills.*``
#: anchor -- and a recorded string that resolves is the only thing a module
#: alias buys.
#:
#: The class names are aliased for source that already imports them.
Datastill = Still
Datalightning = LightningBuilder
Dataweights = Weights
