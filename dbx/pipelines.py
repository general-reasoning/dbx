"""Pipelines: a Datablock tree defined flat, and the journal that finds every pipeline.

`Pipeline` is an abstract Datablock whose one topic, `entrypoint`, is
SYNTHETIC: the root of a tree of blocks it arranges, parceling its own VAR out
among them, in `__create__`. Every creation is recorded in the
`PipeJournal` (in `journals`) -- when its `JOURNAL` says so -- under
`{datalake}/.pipelines`, with the `entrypath` it led to; `dbx.pipejournal`
reads them back, one row per pipeline.

`pipeline` makes a Pipeline of a function that arranges a tree and returns
its root -- an entrypoint function -- its arguments the pipeline's VAR: a
`PipelineFromEntrypointFn`.

`Pipeline.runner` runs a call on the pipeline's entrypoint elsewhere -- a Kueue Job (`dbx.kube`) --
journals it, and lists, watches and cancels its runs: a `dbx.cluster.Runner`.
"""
import dataclasses
import functools
import inspect
import sys
import uuid
from typing import Any

from .databits import default_datalake
from .datablocks import SYNTHETIC, Datablock
from .journals import PipeJournal
from . import cluster as _cluster_
from .cluster import Runner


__all__ = ['Pipeline', 'PipelineFromEntrypointFn', 'pipeline', 'entrypoint_tag']


class Pipeline(Datablock):
    """A Datablock tree, defined flat: one block whose VAR is parceled out among the blocks it arranges.

    A subclass declares the VAR of the whole tree and implements
    `__create__`, which builds the tree out of it -- each block given the
    part of the VAR that is its -- and returns the root: the ENTRYPOINT.

        class Train(Pipeline):
            @dataclass
            class VAR(Pipeline.VAR):
                n: int = 1000
                lr: float = 1e-3

            def __create__(self):
                data = Dataset(spec={'n': self.var.n})
                return Model(spec={'data': data, 'lr': self.var.lr})

        Train().entrypoint.build_tree()

    The one topic, `entrypoint`, is SYNTHETIC: a pipeline stores nothing
    of its own, and is valid vacuously. `read('entrypoint')`,
    `entrypoint` (or `entry`, for short) and `create` all
    return the one tree `create` arranges, once per instance. A
    pipeline whose `JOURNAL` is True records that creation in the
    `PipeJournal`, with the entrypoint's anchorkeypath as `entrypath` --
    which is how the pipelines that led to a block are found again
    (`PipeJournal.collect`).
    """

    TOPICS = {'entrypoint': SYNTHETIC}

    # Whether a creation is recorded in the `PipeJournal`. Off by default:
    # only the pipelines one runs and looks for again -- not the collators,
    # factories and builders they nest -- are worth an entry.
    JOURNAL: bool = False

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
        """The entrypoint `__create__` returns, arranged once per instance; see `_created_`."""
        return self._created_

    # 3. Accessors ---------------------------------------------------------

    @functools.cached_property
    def _created_(self) -> Datablock:
        """The entrypoint `__create__` returns, its creation recorded in the `PipeJournal` as event `'create'` when `JOURNAL`.

        The entrypoint is given this pipeline's `id` as its `pipeline`, which its tree hands down, and
        the `create` entry is written with that `id`: every entry the tree's build writes points to it.
        """
        entrypoint = self.__create__()
        if not isinstance(entrypoint, Datablock):
            raise TypeError(f"{self.__class__.__name__}.__create__() returned a "
                            f"{type(entrypoint).__name__}, not the Datablock at the root of its tree")
        entrypoint._pipeline_ = self.id
        if self.JOURNAL:
            PipeJournal.write(self, 'create', entrypath=entrypoint.anchorkeypath, id=self.id)
        return entrypoint

    @property
    def id(self) -> str:
        """This pipeline instance's id: generated once and kept, as `tree` is, and the `id` of its `create` entry.

        Every block of the tree it creates records it as `pipeline`. Per instance, not per pipeline:
        two instances of one preset -- the same `hash` -- have different ids, as a Job's instance,
        evaluated from the pipeline's quote, has its own.
        """
        if getattr(self, '_id_', None) is None:
            self._id_ = uuid.uuid4().hex[:16] if getattr(self, '_uuid16_', False) else str(uuid.uuid4())
        return self._id_

    @property
    def pipeline(self) -> str:
        """This pipeline's own `id`: a pipeline's entries -- `create`, `job:*` -- name it as its tree's do."""
        return self.id

    @property
    def entrypoint(self) -> Datablock:
        """`create`: the root of this pipeline's tree."""
        return self.create()

    @property
    def entry(self) -> Datablock:
        """`entrypoint`, for short: the root of this pipeline's tree."""
        return self.entrypoint

    @property
    def entrypath(self) -> str:
        """The entrypoint's anchorkeypath."""
        return self.entrypoint.anchorkeypath

    @property
    def operational(self) -> dict:
        """This pipeline's operational keyword arguments: its own, outside its identity.

        What `__create__` hands on to the blocks that run with them -- `n_workers`, `n_gpus`,
        ... -- as `Block(spec=..., **self.operational)`: a keyword a block does not declare stays
        out of its hash too. The datalake is not among them: every block resolves its own.
        """
        explicit = set(self.__explicit_params__())
        return {k: getattr(self, k) for k in self.parameters if k not in explicit and k != 'log'}

    def runner(self, target=None, *, mode: str | None = None, force: bool = False, **options) -> 'Runner':
        """This pipeline's runs on *target*: a `dbx.cluster.Runner` of the target's kind, which launches, lists, watches and cancels them.

            r = p.runner(LAMBDA, n_gpus=8)
            r.build_tree()                           # a Run -- here a dbx.kube.KubeRun
            r.launch('try_specialize_tree', ...)
            r.runs(), r.rundash(), r.cancel()        # its runs on LAMBDA
            p.runner().rundash()                     # its runs on every target

        *target* is where (`dbx.cluster`): a value, e.g. a `dbx.kube.Cluster`, or the name a
        `dbx.clusters` entry point gives one; with none, the Runner lists runs and launches nothing.
        *mode*, when given, is the kind it must be ('kube'). *options* are the kind's --
        `dbx.kube.job`'s for 'kube' -- refused here when it does not take them: among them the pod's
        `n_gpus` (by default the pipeline's `n_gpus`, or 0), `n_cpus` and `mem_gib`; `venv`;
        `workdirs={src: target}` (sent, or packed, before the Job); and `env={name: value}`, the
        Job's environment: over the runner's own `DBX_*` (the datalake, the session, the call),
        which are over the cluster's own `env`. *force* launches `build`/`build_tree` of an entrypoint
        that is valid here.
        """
        return _cluster_.runner(self, target, mode=mode, force=force, **options)


class PipelineFromEntrypointFn(Pipeline):
    """A Pipeline whose tree -- its entrypoint the root -- an entrypoint function arranges; see `pipeline`."""

    # The function arranging the tree; set on each subclass by `pipeline`.
    FUNCTION = None
    # The function's leading arguments, passed back positionally.
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
        ops = super().operational
        if 'datalake' in inspect.signature(type(self).FUNCTION).parameters:
            ops['datalake'] = getattr(self, '_datalake_', None)
        return ops


def _cache_key_(value):
    """*value*, hashable and equal wherever *value* is: dicts, lists, sets and blocks by what they hold."""
    if isinstance(value, Datablock):
        return (type(value), value.quote())
    if isinstance(value, dict):
        return (dict, tuple(sorted(((_cache_key_(k), _cache_key_(v)) for k, v in value.items()), key=repr)))
    if isinstance(value, (list, tuple)):
        return (type(value), tuple(_cache_key_(v) for v in value))
    if isinstance(value, (set, frozenset)):
        return (frozenset, frozenset(_cache_key_(v) for v in value))
    if isinstance(value, type) or callable(value):
        return value
    try:
        hash(value)
    except TypeError:
        return (type(value), repr(value))
    return (type(value), value)


def _camel_(name: str) -> str:
    return ''.join(part[:1].upper() + part[1:] for part in name.split('_'))


def entrypoint_tag(fn, name: str) -> str:
    """The tag of the root block an entrypoint *fn* builds for *name*: `module.qualname:name`.

    *name* selects the preset; the function's qualified name tells apart the pipelines that select by
    the same one. A label to filter journals by -- since dbx keys blocks by 'version_shorthash', not a
    location, so tagging a block that is already built does not move it.

    `pipeline` gives the PIPELINE this same tag when its first argument is *name*; the pipeline does
    not construct the tree, the entrypoint does, so the root's tag is still the entrypoint's to set:

        @pipeline(journal=True)
        def my_clip(name, *, datalake=None):
            return MyClip(tag=entrypoint_tag(my_clip, name), datalake=datalake, spec=...)

    An entrypoint must NOT declare a `tag` parameter of its own: `pipeline` already passes its tag
    to the pipeline class, and a second one arrives in `**ops` -- two values for one keyword.
    """
    return f"{fn.__module__}.{fn.__qualname__}:{name}"


def pipeline(fn=None, *, tag: str | None = None, journal: bool = False):
    """*fn*, returning a `PipelineFromEntrypointFn` rather than the tree it arranges.

    An entrypoint function `f(name, ..., *, var_a=..., var_b=..., n_workers=..., datalake=...)`
    returns the root block of a tree. Its leading arguments -- a configuration's name, a block -- and
    its `var_*` ones are what the tree IS: each lands in some block's VAR. Its other keyword arguments
    say how and where it builds, and land in no identity. `@pipeline` makes it return a pipeline:

        @pipeline
        def deep_feature_clip(name, *, var_cls_token_only=False, n_devices=None, datalake=None):
            ...
            return DeepFeatureClip(...)

        deep_feature_clip('CPTAC_TEST', var_cls_token_only=True).entry.build_tree()

    The pipeline's VAR is the leading arguments, under their names, and the `var_*` ones without the
    prefix; the operational ones are its own keyword arguments, kept out of its identity, as a
    Datastack's `n_workers` is. Its `__create__` calls the function back with all of them -- so the
    function is the one definition of the tree, and the pipeline its name, its configuration, and its
    record in the `PipeJournal`. Within a function, another entrypoint's block is held as `g(...).entry`:
    a block held directly hashes as its quote does. A `**kwargs` of the function's is operational:
    passed through, never a VAR, and refusing a `var_*`. `datalake=` is every pipeline's -- where
    it is journaled -- and passed to the function only when the function takes one.

    `@pipeline` or `@pipeline(tag=...)`: each pipeline is tagged *tag*, by default the function's
    fully qualified name, `module.qualname` -- several projects may write to one journal -- to filter
    the `PipeJournal` by. When the function's first argument is a `name` (a preset's), the call's
    tag is `<tag>:<name>`, as `entrypoint_tag` makes it -- so the pipelines of one function are told
    apart by their presets, and a pipeline and its root carry one tag. A tag filter matches by
    containment, so the function's own name, or the preset's, finds it too.
    `@pipeline(journal=True)` sets the pipeline's `Pipeline.JOURNAL`: only then is its creation
    journaled. Leave it off for the minor pipelines -- collators, factories -- others nest.

    This tag is the PIPELINE's. The root block the function returns is tagged by the function, with
    `entrypoint_tag` -- the same string -- so an entrypoint takes no `tag` parameter of its own,
    which would reach the pipeline class twice.

    A call is cached by ALL its arguments, as `functools.cache` would cache it, for the life of the
    process: the same arguments return the same pipeline, so a pipeline that several others nest is
    created -- and journaled -- once. Arguments `functools.cache` would refuse -- dicts, lists, sets,
    blocks -- are keyed by value, a block by its quote -- and an omitted `datalake=` by the
    `DBX_DATALAKE` it stands for at the call. `f.cache_clear()` empties it.
    The pipeline class, `<FnInCamelCase>Pipeline`, is put in the function's module under that name,
    so its quote -- the journal's record -- evaluates. `fn(None)`, the block class some entrypoints
    answer with, is passed through.
    """
    if fn is None:
        return functools.partial(pipeline, tag=tag, journal=journal)
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
        'VAR': VAR, 'FUNCTION': staticmethod(fn), 'LEADING': leading, 'JOURNAL': journal, '__module__': fn.__module__,
        '__qualname__': name, '__doc__': f"The pipeline of `{fn.__name__}`: its tree, configured by its VAR.",
    })
    VAR.__qualname__ = f"{name}.VAR"
    setattr(sys.modules[fn.__module__], name, cls)

    cache = {}

    @functools.wraps(fn)
    def entry(*args, **kwargs):
        # An omitted datalake= is DBX_DATALAKE's as of the call, which a block resolves as it is constructed.
        key = _cache_key_((args, kwargs, default_datalake()))
        if key not in cache:
            cache[key] = arrange(*args, **kwargs)
        return cache[key]

    def arrange(*args, **kwargs):
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
        name = spec.get('name') if leading and leading[0] == 'name' else None
        return cls(spec=spec, datalake=datalake, tag=f"{tag}:{name}" if isinstance(name, str) else tag, **ops)

    entry.Pipeline = cls
    entry.function = fn
    entry.cache_clear = cache.clear
    return entry

