import inspect
from dataclasses import dataclass, fields, is_dataclass

from .datablocks import Datablock, LogVolume


class Datablockable:
    """Minimal base providing path infrastructure.

    Subclasses that define ``TOPICFILES`` (a ``{topic: filename_or_None}``
    dict) can call ``self.path(topic)`` to resolve storage paths without
    importing ``dbx``.

    Instances are typically created directly for testing or passed through
    ``dbx.datablock()`` in pipeline modules for full dbx integration.
    """

    TOPICFILES = {}
    
    @dataclass
    class CONFIG:
        ...   

    def __init__(self, *, root, anchor, key, cfg, log_volume, log, device):
        self.root = root
        self.anchor = anchor
        self.key = key
        self.cfg = cfg
        self.log_volume = log_volume
        self.log = log
        self.device = device

    def build(self):
        return self.__pre_build__().__build__().__post_build__()

    def __pre_build__(self):
        return self

    def __build__(self):
        return self

    def __post_build__(self):
        return self

    def read(self, topic=None):
        if topic not in self.TOPICFILES:
            raise ValueError(f"Topic {repr(topic)} not in {self.TOPICFILES}")
        return self.__read__(topic)
    
    def __read__(self, topic=None):
        raise NotImplementedError()

    @property
    def anchorkeypath(self):
        return os.path.join(self.root, self.anchor, self.key)

    def path(self, topic=None, *, ensure_dirpath=False):
        """Return the storage path for *topic*.

        When ``TOPICFILES[topic]`` is a filename string the path is::

            root / anchor / key / topic / filename

        When ``TOPICFILES[topic]`` is ``None`` the path is the topic
        directory itself::

            root / anchor / key / topic

        When *topic* is ``None`` the key-level directory is returned::

            root / anchor / key

        If *ensure_dirpath* is ``True`` the directory containing the
        returned path is created (``os.makedirs``) before returning.
        """
        if topic is not None:
            topicfile = self.TOPICFILES[topic]
            if topicfile is not None:
                p = os.path.join(
                    self.anchorkeypath, topic, topicfile,
                )
            else:
                p = os.path.join(
                    self.anchorkeypath, topic,
                )
        else:
            p = self.anchorkeypath

        if ensure_dirpath:
            os.makedirs(p if topic is None or self.TOPICFILES.get(topic) is None
                        else os.path.dirname(p),
                        exist_ok=True)
        return p


class Datastackable(Datablockable):
    """Minimal base for classes that produce a stack of shards.

    Extends :class:`Datablockable` with two additional protocol methods
    required by ``dbx.datastack()``:

    ``shard(idx)``
        Return the :class:`Datablockable` for shard *idx*.  The returned
        object must itself define ``build()`` and ``read()``.

    ``stack()``
        Called once after all shards have been built.  Produces any
        stack-level artefacts (e.g. consolidated index, manifests).

    ``build()``
        Default implementation builds every shard sequentially via
        ``shard(idx).build()`` then calls ``stack()``.  Override when
        parallel or custom orchestration is needed.
    """

    def __build__(self):
        for shard in self.shards():
            shard.build()
        return self.__stack__()
    
    def shard(self, idx: int):
        """Return the shard at *idx*, potentially lazily forming ``_shards_`` if needed."""
        return self.__shard__(idx)

    @property
    def n_shards(self):
        return 0

    def shards(self) -> list:
        """Return all shards, forming them via :meth:`shard` if needed."""
        return [self.shard(idx) for idx in range(self.n_shards)]

    def __shard__(self, idx: int):
        """Return a single child :class:`Datablock` for the given index.

        Subclasses **must** override this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement __shard__(idx)"
        )

    def __stack__(self):
        """Produce stack-level artefacts after all shards are built."""
        return self


# ---------------------------------------------------------------------------
# CONFIG lifting helper (shared by datablock() and datastack())
# ---------------------------------------------------------------------------

def _lift_config(cls, class_attrs):
    """Lift CONFIG from *cls* into *class_attrs*, ensuring it inherits from
    ``Datablock.CONFIG`` so that the LazyLoader / specline mechanism works.
    """
    if hasattr(cls, 'CONFIG') and is_dataclass(cls.CONFIG):
        user_config = cls.CONFIG
        if issubclass(user_config, Datablock.CONFIG):
            class_attrs['CONFIG'] = user_config
        else:
            from dataclasses import MISSING, field as _dc_field_
            ns = {'__annotations__': {}}
            for f in fields(user_config):
                ns['__annotations__'][f.name] = f.type
                if f.default is not MISSING:
                    ns[f.name] = f.default
                elif f.default_factory is not MISSING:
                    ns[f.name] = _dc_field_(default_factory=f.default_factory)
            new_config = dataclass(
                type(user_config.__name__, (Datablock.CONFIG,), ns)
            )
            class_attrs['CONFIG'] = new_config
    else:
        @dataclass
        class _EmptyCONFIG(Datablock.CONFIG):
            pass
        class_attrs['CONFIG'] = _EmptyCONFIG


# ---------------------------------------------------------------------------
# Class attribute lifting helper
# ---------------------------------------------------------------------------

def _lift_class_attrs(cls, class_attrs):
    """Lift TOPICFILES, TOPICFILE, VERSION, and CONFIG from *cls*."""
    if hasattr(cls, 'TOPICFILES'):
        class_attrs['TOPICFILES'] = cls.TOPICFILES
    if hasattr(cls, 'TOPICFILE'):
        class_attrs['TOPICFILE'] = cls.TOPICFILE
    if hasattr(cls, 'VERSION'):
        class_attrs['VERSION'] = cls.VERSION
    _lift_config(cls, class_attrs)


# ===========================================================================
# datablock()
# ===========================================================================


def datablock(cls, *, underride=('build', 'read')):
    """Wrap a Datablockable class as a Datablock subclass.

    The wrapper creates a dynamic class ``(cls, Datablock)`` and
    *underrides* selected methods: the framework version (from
    ``Datablock``) is placed into the wrapper's ``__dict__`` so it takes
    priority over ``cls``'s version in the MRO.  All other methods
    resolve normally via MRO (``cls`` first, then ``Datablock``).

    ``__init__`` is always underridden with ``Datablock.__init__``.

    Parameters
    ----------
    cls : type
        A class satisfying the Datablockable protocol.
    underride : list[str], optional
        Method names to underride with ``Datablock`` versions.
        Default: ``['build', 'read']``.

    Returns
    -------
    type
        A dynamically-created ``Datablock`` subclass wrapping *cls*.

    Usage::

        FeatureBlock = dbx.datablock(FeatureExtractor)
        FeatureBlock = dbx.datablock(FeatureExtractor, underride=['build'])
    """
    # -- Validate protocol --------------------------------------------------------
    if not hasattr(cls, '__build__'):
        raise TypeError(f"{cls.__name__} must define __build__() to be Datablockable")
    if not hasattr(cls, '__read__'):
        raise TypeError(f"{cls.__name__} must define __read__() to be Datablockable")

    # -- Collect class-level attributes -------------------------------------------
    class_attrs = {}
    _lift_class_attrs(cls, class_attrs)

    # -- Always underride __init__ ------------------------------------------------
    class_attrs['__init__'] = Datablock.__init__

    # -- Underride specified methods ----------------------------------------------
    for method_name in underride:
        framework_method = getattr(Datablock, method_name, None)
        if framework_method is not None:
            class_attrs[method_name] = framework_method

    # -- Pickling support ---------------------------------------------------------
    def __reduce__(self):
        return (_unpickle_datablock_instance,
                (cls, underride, self.__class__.__module__, self.__getstate__()))

    class_attrs['__reduce__'] = __reduce__

    # -- from_datablockable classmethod -------------------------------------------
    @classmethod
    def from_datablockable(wrapper_cls, obj, *, root=None, **kwargs):
        """Create a Datablock wrapper from an existing datablockable instance.

        Extracts the CONFIG fields from *obj*.cfg as the ``spec`` dict,
        and propagates ``log_volume`` from the datablockable instance
        to the wrapper.
        """
        if not isinstance(obj, cls):
            raise TypeError(
                f"Expected an instance of {cls.__name__}, "
                f"got {type(obj).__name__}"
            )
        cfg = obj.cfg
        if is_dataclass(cfg):
            spec = {f.name: getattr(cfg, f.name) for f in fields(cfg)}
        elif hasattr(cfg, '__dict__'):
            spec = dict(cfg.__dict__)
        else:
            raise TypeError(
                f"Cannot extract spec from {type(cfg).__name__}: "
                f"expected a dataclass or object with __dict__"
            )
        for attr in ('tag',):
            if attr not in kwargs and hasattr(obj, attr):
                kwargs[attr] = getattr(obj, attr)
        if 'log_volume' not in kwargs:
            lv = {}
            for attr in ('info', 'verbose', 'debug', 'detailed'):
                if hasattr(obj, attr):
                    lv[attr] = getattr(obj, attr)
            if lv:
                kwargs.update(lv)
        return wrapper_cls(root=root, spec=spec, **kwargs)

    class_attrs['from_datablockable'] = from_datablockable

    # -- Create the subclass dynamically ------------------------------------------
    wrapper_name = f'{cls.__name__}_Datablock'

    caller_module = inspect.currentframe().f_back.f_globals.get(
        '__name__', cls.__module__
    )

    WrapperClass = type(wrapper_name, (cls, Datablock), class_attrs)
    WrapperClass.__module__ = caller_module
    WrapperClass.__qualname__ = wrapper_name
    WrapperClass.__wrapped__ = cls

    return WrapperClass


def _unpickle_datablock_instance(cls, underride, module, state):
    """Reconstruct a ``datablock(cls)`` wrapper instance from pickled state."""
    WrapperClass = datablock(cls, underride=underride)
    WrapperClass.__module__ = module
    obj = WrapperClass.__new__(WrapperClass)
    obj.__setstate__(state)
    return obj


# ===========================================================================
# datastack()
# ===========================================================================


def datastack(cls, *, underride=('build', 'read', 'shard')):
    """Wrap a Datastackable class as a :class:`Datastack` subclass.

    The wrapper creates a dynamic class ``(cls, Datastack)`` and
    *underrides* selected methods: the framework version is placed into
    the wrapper's ``__dict__`` so it takes priority over ``cls``'s version
    in the MRO.  All other methods resolve normally via MRO.

    ``__init__`` is always underridden with ``Datastack.__init__``.

    Additionally, a ``__shard__`` conversion layer is always installed:
    it wraps the raw Datablockable returned by ``cls.__shard__()`` into a
    proper ``Datablock`` via ``from_datablockable()``.

    Parameters
    ----------
    cls : type
        A class satisfying the Datastackable protocol.
    underride : list[str], optional
        Method names to underride with ``Datastack``/``Datablock`` versions.
        Default: ``['build', 'read', 'shard']``.

    Returns
    -------
    type
        A dynamically-created ``Datastack`` subclass wrapping *cls*.

    Usage::

        MyStack = dbx.datastack(MyPipeline)
        MyStack = dbx.datastack(MyPipeline, underride=['build', 'shard'])
    """
    from .datablocks import Datastack

    # -- Validate protocol --------------------------------------------------------
    if not hasattr(cls, '__shard__'):
        raise TypeError(f"{cls.__name__} must define __shard__() to be Datastackable")
    if not hasattr(cls, 'n_shards'):
        raise TypeError(f"{cls.__name__} must define n_shards to be Datastackable")
    if not hasattr(cls, 'SHARD'):
        raise TypeError(f"{cls.__name__} must define SHARD to be Datastackable")

    # -- Create the Datablock wrapper for this stack's shard class ----------------
    ShardBlock = datablock(cls.SHARD)

    # -- Collect class-level attributes -------------------------------------------
    class_attrs = {}
    class_attrs['_ShardBlock_'] = ShardBlock
    _lift_class_attrs(cls, class_attrs)

    # -- Always underride __init__ ------------------------------------------------
    class_attrs['__init__'] = Datastack.__init__

    # -- Underride specified methods ----------------------------------------------
    for method_name in underride:
        # Look up from Datastack first, falling back to Datablock.
        framework_method = getattr(Datastack, method_name, None)
        if framework_method is None:
            framework_method = getattr(Datablock, method_name, None)
        if framework_method is not None:
            class_attrs[method_name] = framework_method

    # -- __shard__ conversion layer -----------------------------------------------
    # cls.__shard__() returns a raw Datablockable, but the framework needs
    # Datablock instances.  This wrapper converts via from_datablockable().
    def __shard__(self, idx: int):
        datablockable_shard = cls.__shard__(self, idx)
        return self._ShardBlock_.from_datablockable(
            datablockable_shard,
            root=self.root,
            anchor=self.anchor,
            keyby=self.keyby,
        )
    class_attrs['__shard__'] = __shard__

    # -- Pickling support ---------------------------------------------------------
    def __reduce__(self):
        return (_unpickle_datastack_instance,
                (cls, underride, self.__class__.__module__, self.__getstate__()))

    class_attrs['__reduce__'] = __reduce__

    # -- from_datastackable classmethod -------------------------------------------
    @classmethod
    def from_datastackable(wrapper_cls, obj, *, root=None, **kwargs):
        """Create a Datastack wrapper from an existing Datastackable instance."""
        if not isinstance(obj, cls):
            raise TypeError(
                f"Expected an instance of {cls.__name__}, "
                f"got {type(obj).__name__}"
            )
        cfg = obj.cfg
        if is_dataclass(cfg):
            spec = {f.name: getattr(cfg, f.name) for f in fields(cfg)}
        elif hasattr(cfg, '__dict__'):
            spec = dict(cfg.__dict__)
        else:
            raise TypeError(
                f"Cannot extract spec from {type(cfg).__name__}: "
                f"expected a dataclass or object with __dict__"
            )
        if 'log_volume' not in kwargs:
            lv = {}
            for attr in ('info', 'verbose', 'debug', 'detailed'):
                if hasattr(obj, attr):
                    lv[attr] = getattr(obj, attr)
            if lv:
                kwargs.update(lv)
        return wrapper_cls(root=root, spec=spec, **kwargs)

    class_attrs['from_datastackable'] = from_datastackable

    # -- Create the subclass dynamically ------------------------------------------
    wrapper_name = f'{cls.__name__}_Datastack'

    caller_module = inspect.currentframe().f_back.f_globals.get(
        '__name__', cls.__module__
    )

    WrapperClass = type(wrapper_name, (cls, Datastack), class_attrs)
    WrapperClass.__module__ = caller_module
    WrapperClass.__qualname__ = wrapper_name
    WrapperClass.__wrapped__ = cls

    ShardBlock.__module__ = caller_module

    return WrapperClass


def _unpickle_datastack_instance(cls, underride, module, state):
    """Reconstruct a ``datastack(cls)`` wrapper instance from pickled state."""
    WrapperClass = datastack(cls, underride=underride)
    WrapperClass.__module__ = module
    obj = WrapperClass.__new__(WrapperClass)
    obj.__setstate__(state)
    return obj
