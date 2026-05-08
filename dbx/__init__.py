"""dbx — Content-addressed, journaled data-experiment management.

The ``dbx`` package provides :class:`~dbx.datablocks.Datablock` and
:class:`~dbx.datablocks.Datastack`, content-addressed building blocks for
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
"""

__version__ = "0.0.1"

from .dataparts import *
from .datablocks import *