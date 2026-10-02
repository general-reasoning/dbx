"""dbx — Config-addressed, journaled data-experiment management.

The ``dbx`` package provides :class:`~dbx.datablocks.Datablock` and
:class:`~dbx.datablocks.Datastack`, config-addressed building blocks for
organising data pipelines.  Every block is identified by a deterministic
hash of its configuration; every build is journaled for full
reproducibility.

Quick start::

    from dbx import Datablock, Datastack

Key modules
-----------
dataparts
    Standalone utilities: I/O helpers (``read_frame``,
    ``write_tensor``, …), callable executors for threading / multiprocessing /
    Ray parallelism.
datablocks
    :class:`Datablock`, :class:`Datastack`, git-revision tracking, remote
    execution via Ray.
journals
    :class:`Datajournal` and the build records it reads and writes
    (:class:`DatajournalFrame`, :class:`DatajournalEntry`); the exec journal
    (:func:`execjournal`, :class:`ExecjournalFrame`); the filters both are
    queried with; and the log, :class:`Datalog`.

Modules NOT imported here
-------------------------
``import dbx`` must work without torch, so two modules are imported by name
instead -- ``from dbx.datastreams import ...``, ``from dbx.stills import
...``:

datastreams
    Shard-backed dataset/loader plumbing: :class:`ZipStreamingDataset`,
    :class:`ChunkShuffleSampler`, :class:`ResumableDataLoader`,
    ``chunk_split_indices``, ``val_loader_workers``. Needs torch, and
    mosaicml-streaming for the MDS parts.
stills
    One training run as a Datablock: :class:`Still`,
    :class:`LightningBuilder`, :class:`ModelBuilder`, :class:`DatasetBuilder`,
    :class:`CheckpointBuilder`, :class:`Weights`. Needs
    lightning.
"""

__version__ = "0.0.1"

from .dataparts import *
from .datablocks import *
from .journals import *
from .datatables import *
from .backbones import *
from .featuretables import *
from .probes import *

