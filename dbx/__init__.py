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
    Standalone utilities: :class:`Logger`, I/O helpers (``read_frame``,
    ``write_tensor``, …), callable executors for threading / multiprocessing /
    Ray parallelism.
datablocks
    :class:`Datablock`, :class:`Datastack`, journaling, git-revision
    tracking, remote execution via Ray.

Modules NOT imported here
-------------------------
``import dbx`` must work without torch, so two modules are imported by name
instead -- ``from dbx.datastreams import ...``, ``from dbx.datastills import
...``:

datastreams
    Shard-backed dataset/loader plumbing: :class:`ZipStreamingDataset`,
    :class:`ChunkShuffleSampler`, :class:`ResumableDataLoader`,
    ``block_split_indices``, ``val_loader_workers``. Needs torch, and
    mosaicml-streaming for the MDS parts.
datastills
    One training run as a Datablock: :class:`Datastill`,
    :class:`Datalightning`, :class:`Dataweights`, ``scaffold_still``. Needs
    lightning.
"""

__version__ = "0.0.1"

from .dataparts import *
from .datablocks import *
from .datatables import *
from .databackbones import *
from .featuretables import *
from .dataprobes import *

# Backward compatibility alias
import sys
from . import databackbones as datamodels
sys.modules['dbx.datamodels'] = datamodels

# The names datatables and featuretables used to have, aliased the same way.
#
# Load-bearing rather than a courtesy to importers: a class's `fqcn` is its
# `__module__` plus its name, `anchor` falls back to `fqcn`, and both
# `anchorkeypath` and the journal directory are built from that -- so these
# strings are recorded in artifacts on disk. `quotefn` renders `__module__`
# into a specline too, and a specline stands in a spec as text, hence in a
# hash. See the note at the foot of dbx/datatables.py.
#
# The SAME module object under two names, never a shim that forwards to it. A
# forwarder has a namespace of its own, so
# `monkeypatch.setattr(dbx.datapoints, 'StreamingDataset', ...)` would patch
# the forwarder while the code under test read the name out of its own module
# -- leaving the patch a silent no-op, and a test that believed it had stubbed
# a remote read performing one instead. One object cannot drift from itself.
from . import datatables as datapoints
from . import featuretables as datafeatures
sys.modules['dbx.datapoints'] = datapoints
sys.modules['dbx.datafeatures'] = datafeatures