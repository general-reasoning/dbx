"""dbx — Config-addressed, journaled data-experiment management.

The `dbx` package provides `Datablock` and
`Datastack`, config-addressed building blocks for
organising data pipelines.  Every block is identified by a deterministic
hash of its configuration; every build is journaled for full
reproducibility.

Quick start:

    from dbx import Datablock, Datastack

Key modules
-----------
databits
    Standalone utilities: I/O helpers (`read_frame`,
    `write_tensor`, …), callable executors for threading / multiprocessing /
    Ray parallelism.
datablocks
    `Datablock`, `Datastack`, git-revision tracking, remote
    execution via Ray.
datatables
    `Datatablet`, `Datatable` and `Datacollator`: tables of
    MDS slices, and the columns a consumer reads from them.
dataparts
    `DatatablePartition` and `DatatableClusterCorePartition`: a table
    assigned into parts; `PartitionDatamapTable`, a part's map; and
    `MappedDatatable`, a table pulled back along a map -- a part.
journals
    `BuildJournal` and the build records it reads and writes
    (`BuildJournalFrame`, `BuildJournalEntry`); the exec journal
    (`execjournal`, `ExecJournalFrame`); the filters both are
    queried with; and the log, `Datalog`.
pipelines
    `Pipeline`, a Datablock tree defined flat; every creation of
    one is recorded in the `PipeJournal`, read with
    `pipejournal`.

Modules NOT imported here
-------------------------
`import dbx` must work without torch, so two modules are imported by name
instead -- `from dbx.datastreams import ...`, `from dbx.stills import
...`:

datastreams
    Shard-backed dataset/loader plumbing: `ZipStreamingDataset`,
    `ChunkShuffleSampler`, `ResumableDataLoader`,
    `chunk_split_indices`, `val_loader_workers`. Needs torch, and
    mosaicml-streaming for the MDS parts.
stills
    One training run as a Datablock: `Still`,
    `LightningBuilder`, `ModelBuilder`, `DatasetBuilder`,
    `CheckpointBuilder`, `Weights`. Needs
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

