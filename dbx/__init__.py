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
databits
    Standalone utilities: I/O helpers (``read_frame``,
    ``write_tensor``, …), callable executors for threading / multiprocessing /
    Ray parallelism.
datablocks
    :class:`Datablock`, :class:`Datastack`, git-revision tracking, remote
    execution via Ray.
datatables
    :class:`Datatablet`, :class:`Datatable` and :class:`Datacollator`: tables of
    MDS slices, and the columns a consumer reads from them.
dataparts
    :class:`DatatablePartition` and :class:`DatatableCorePartition`: a table
    assigned into parts; :class:`PartitionDatamapTable`, a part's map; and
    :class:`MappedDatatable`, a table pulled back along a map -- a part.
journals
    :class:`DataJournal` and the build records it reads and writes
    (:class:`DataJournalFrame`, :class:`DataJournalEntry`); the exec journal
    (:func:`execjournal`, :class:`ExecJournalFrame`); the filters both are
    queried with; and the log, :class:`Datalog`.
pipelines
    :class:`Pipeline`, a Datablock tree defined flat; every creation of
    one is recorded in the :class:`PipeJournal`, read with
    :func:`pipejournal`.

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

import importlib.util as _importlib_util

from .databits import *
from .datablocks import *
from .journals import *

# The data modules stand on dbx.datastreams, which needs torch and
# mosaicml-streaming -- the [streaming] extra -- at import. They are imported
# only when both are installed, so that `import dbx` works without them:
# reading a journal needs neither. The packages are looked up rather than an
# ImportError caught, so that a real error in one of these modules is not
# mistaken for a missing extra and swallowed.
if _importlib_util.find_spec('torch') and _importlib_util.find_spec('streaming'):
    from .datatables import *
    from .dataparts import *
    from .backbones import *
    from .featuretables import *
    from .probes import *
from .pipelines import *

