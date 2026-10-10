"""Pipelines: a Datablock tree defined flat, and the journal that finds every pipeline.

:class:`Pipeline` is an abstract Datablock whose one topic, ``entrypoint``, is
SYNTHETIC: the root of a tree of blocks it arranges, parceling its own VAR out
among them, in ``__create__``. Every creation is recorded in the
`PipeJournal` (in ``journals``), under ``{datalake}/.pipelines``, with the
``entrypath`` it led to; :func:`dbx.pipejournal` reads them back, one row per
pipeline.
"""
from .datablocks import SYNTHETIC, Datablock
from .journals import PipeJournal


__all__ = ['Pipeline']


class Pipeline(Datablock):
    """A Datablock tree, defined flat: one block whose VAR is parceled out among the blocks it arranges.

    A subclass declares the VAR of the whole tree and implements
    ``__create__``, which builds the tree out of it -- each block given the
    part of the VAR that is its -- and returns the root: the ENTRYPOINT. ::

        class Train(Pipeline):
            @dataclass
            class VAR(Pipeline.VAR):
                n: int = 1000
                lr: float = 1e-3

            def __create__(self):
                data = Dataset(spec={'n': self.var.n})
                return Model(spec={'data': data, 'lr': self.var.lr})

        Train().entrypoint.build_tree()

    The one topic, ``entrypoint``, is SYNTHETIC: a pipeline stores nothing
    of its own, and is valid vacuously. ``read('entrypoint')``,
    :attr:`entrypoint` and :meth:`create` all create the tree afresh,
    and each records the creation in the `PipeJournal`, with the
    entrypoint's anchorkeypath as ``entrypath`` -- which is how the pipelines
    that led to a block are found again (`PipeJournal.collect`).
    """

    TOPICS = {'entrypoint': SYNTHETIC}

    # 1. Datablock protocol ------------------------------------------------

    def __create__(self) -> Datablock:
        """Arrange this pipeline's tree out of its VAR, and return its root."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement __create__(): arrange the tree of blocks "
            f"this pipeline's VAR is parceled out among, and return its root"
        )

    def __read__(self, topic='entrypoint'):
        return self.create()

    # 2. Declared API ------------------------------------------------------

    def create(self) -> Datablock:
        """The entrypoint ``__create__`` returns, its creation recorded in the `PipeJournal` as event ``'create'``."""
        entrypoint = self.__create__()
        if not isinstance(entrypoint, Datablock):
            raise TypeError(f"{self.__class__.__name__}.__create__() returned a "
                            f"{type(entrypoint).__name__}, not the Datablock at the root of its tree")
        PipeJournal.write(self, 'create', entrypath=entrypoint.anchorkeypath)
        return entrypoint

    # 3. Accessors ---------------------------------------------------------

    @property
    def entrypoint(self) -> Datablock:
        """:meth:`create`: the root of this pipeline's tree, created afresh."""
        return self.create()

    @property
    def entrypath(self) -> str:
        """The entrypoint's anchorkeypath."""
        return self.entrypoint.anchorkeypath
