"""Pipelines: a Datablock tree defined flat, and the journal that finds every pipeline.

:class:`Pipeline` is an abstract Datablock whose one topic, ``entrypoint``, is
SYNTHETIC: the root of a tree of blocks it arranges, parceling its own VAR out
among them, in ``__create__``. Every creation is recorded in the
`PipeJournal` (in ``journals``), under ``{datalake}/.pipelines``, with the
``entrypath`` it led to; :func:`dbx.pipejournal` reads them back, one row per
pipeline.

:func:`pipeline` makes a Pipeline of a function that arranges a tree and returns
its root -- an entrypoint function -- its arguments the pipeline's VAR: a
:class:`PipelineFromEntrypointFn`.
"""
import dataclasses
import functools
import inspect
import sys
from typing import Any

from .datablocks import SYNTHETIC, Datablock
from .journals import PipeJournal


__all__ = ['Pipeline', 'PipelineFromEntrypointFn', 'pipeline']


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
    :attr:`entrypoint` (or :attr:`entry`, for short) and :meth:`create` all
    create the tree afresh,
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
    def entry(self) -> Datablock:
        """:attr:`entrypoint`, for short: the root of this pipeline's tree, created afresh."""
        return self.entrypoint

    @property
    def entrypath(self) -> str:
        """The entrypoint's anchorkeypath."""
        return self.entrypoint.anchorkeypath


class PipelineFromEntrypointFn(Pipeline):
    """A Pipeline whose tree -- its entrypoint the root -- an entrypoint function arranges; see `pipeline`."""

    #: The function arranging the tree; set on each subclass by `pipeline`.
    FUNCTION = None
    #: The function's leading arguments, passed back positionally.
    LEADING: tuple = ()

    def __create__(self):
        var = self.var
        args = [getattr(var, a) for a in self.LEADING]
        kwargs = {f'var_{f.name}': getattr(var, f.name) for f in dataclasses.fields(var)
                  if f.name not in self.LEADING}
        return type(self).FUNCTION(*args, **kwargs, **self.operational)

    @property
    def operational(self) -> dict:
        """The function's operational keyword arguments: this block's own, outside its identity, and its datalake."""
        explicit = set(self.__explicit_params__())
        ops = {k: getattr(self, k) for k in self.parameters if k not in explicit and k != 'log'}
        if 'datalake' in inspect.signature(type(self).FUNCTION).parameters:
            ops['datalake'] = getattr(self, '_datalake_', None)
        return ops


def _camel_(name: str) -> str:
    return ''.join(part[:1].upper() + part[1:] for part in name.split('_'))


def pipeline(fn=None, *, tag: str | None = None):
    """*fn*, returning a `PipelineFromEntrypointFn` rather than the tree it arranges.

    An entrypoint function ``f(name, ..., *, var_a=..., var_b=..., n_workers=..., datalake=...)``
    returns the root block of a tree. Its leading arguments -- a configuration's name, a block -- and
    its ``var_*`` ones are what the tree IS: each lands in some block's VAR. Its other keyword arguments
    say how and where it builds, and land in no identity. ``@pipeline`` makes it return a pipeline::

        @pipeline
        def deep_feature_clip(name, *, var_cls_token_only=False, n_devices=None, datalake=None):
            ...
            return DeepFeatureClip(...)

        deep_feature_clip('CPTAC_TEST', var_cls_token_only=True).entry.build_tree()

    The pipeline's VAR is the leading arguments, under their names, and the ``var_*`` ones without the
    prefix; the operational ones are its own keyword arguments, kept out of its identity, as a
    Datastack's ``n_workers`` is. Its ``__create__`` calls the function back with all of them -- so the
    function is the one definition of the tree, and the pipeline its name, its configuration, and its
    record in the `PipeJournal`. Within a function, another entrypoint's block is held as ``g(...).entry``:
    a block held directly hashes as its quote does. A ``**kwargs`` of the function's is operational:
    passed through, never a VAR, and refusing a ``var_*``. ``datalake=`` is every pipeline's -- where
    it is journaled -- and passed to the function only when the function takes one.

    ``@pipeline`` or ``@pipeline(tag=...)``: each pipeline is tagged *tag*, by default the function's
    fully qualified name, ``module.qualname`` -- several projects may write to one journal -- to filter
    the `PipeJournal` by; a tag filter matches by containment, so the function's own name finds it too.
    The pipeline class, ``<FnInCamelCase>Pipeline``, is put in the function's module under that name,
    so its quote -- the journal's record -- evaluates. ``fn(None)``, the block class some entrypoints
    answer with, is passed through.
    """
    if fn is None:
        return functools.partial(pipeline, tag=tag)
    tag = f"{fn.__module__}.{fn.__qualname__}" if tag is None else tag
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    leading = tuple(p.name for p in params
                    if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD))
    var_params = [p for p in params if p.kind is inspect.Parameter.KEYWORD_ONLY and p.name.startswith('var_')]

    def var_field(p):
        name = p.name[len('var_'):] if p.name.startswith('var_') else p.name
        if p.default is inspect.Parameter.empty:
            return (name, Any)
        if isinstance(p.default, (list, dict, set)):
            default = p.default
            return (name, Any, dataclasses.field(default_factory=lambda: type(default)(default)))
        return (name, Any, dataclasses.field(default=p.default))

    VAR = dataclasses.make_dataclass(
        'VAR', [var_field(p) for p in params if p.name in leading] + [var_field(p) for p in var_params],
        bases=(Pipeline.VAR,), kw_only=True)
    name = _camel_(fn.__name__) + 'Pipeline'
    cls = type(name, (PipelineFromEntrypointFn,), {
        'VAR': VAR, 'FUNCTION': staticmethod(fn), 'LEADING': leading, '__module__': fn.__module__,
        '__qualname__': name, '__doc__': f"The pipeline of `{fn.__name__}`: its tree, configured by its VAR.",
    })
    VAR.__qualname__ = f"{name}.VAR"
    setattr(sys.modules[fn.__module__], name, cls)

    @functools.wraps(fn)
    def entry(*args, **kwargs):
        # The pipeline's datalake, where it is journaled, given to fn only when fn takes one.
        datalake = None if 'datalake' in sig.parameters else kwargs.pop('datalake', None)
        bound = sig.bind(*args, **kwargs)
        if leading and leading[0] == 'name' and bound.arguments.get('name', ...) is None:
            return fn(*args, **kwargs)
        spec, ops = {}, {}
        for k, v in bound.arguments.items():
            if sig.parameters[k].kind is inspect.Parameter.VAR_KEYWORD:
                # A **kwargs passes operational arguments through: a var_* among them would be no field.
                given = sorted(n for n in v if n.startswith('var_'))
                if given:
                    raise TypeError(f"{fn.__name__}: {given} would be identity, and its **{k} is operational only")
                ops.update(v)
            elif k in leading:
                spec[k] = v
            elif k.startswith('var_'):
                spec[k[len('var_'):]] = v
            else:
                ops[k] = v
        datalake = ops.pop('datalake', datalake)
        return cls(spec=spec, datalake=datalake, tag=tag, **ops)

    entry.Pipeline = cls
    entry.function = fn
    return entry

