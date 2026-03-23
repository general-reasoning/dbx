import inspect
from dataclasses import dataclass, fields, is_dataclass

from .datablocks import Datablock


def datablock(cls):
    """Wrap a Datablockable class as a Datablock subclass.

    A class is *Datablockable* if it defines:

        TOPICFILES = {topic: filename, ...}   # or TOPICFILE = 'filename'
        VERSION = '...'                       # optional

        @dataclass
        class CONFIG:
            ...

        def __init__(self, *, cfg, verbose, detailed, debug, log):
            ...

        def __build__(self, *args, **kwargs):
            ...
            return self

        def __read__(self, topic):
            ...

    Optionally, a Datablockable may define:

        def path(self, topic=None, *, ensure_dirpath=False):
            # Override Datablock.path() with custom path logic
            ...

    Usage::

        FeatureBlock = dbx.datablock(FeatureExtractor)
        block = FeatureBlock(root='/data', spec={'model': 'resnet50'})
        block.build()
        result = block.read('features')

    The returned class is a proper ``Datablock`` subclass named
    ``_<cls.__name__>_Datablock_``.  It can also be used as a decorator::

        @dbx.datablock
        class MyProcessor:
            ...

    Parameters
    ----------
    cls : type
        A class satisfying the Datablockable protocol.

    Returns
    -------
    type
        A dynamically-created ``Datablock`` subclass wrapping *cls*.
    """
    # -- Validate protocol --------------------------------------------------------
    if not hasattr(cls, '__build__'):
        raise TypeError(f"{cls.__name__} must define __build__ to be Datablockable")
    if not hasattr(cls, '__read__'):
        raise TypeError(f"{cls.__name__} must define __read__ to be Datablockable")
    # NOTE: TOPICFILES / TOPICFILE may be defined at class level OR in __init__.
    # We lift class-level ones here; instance-level ones are propagated in
    # __post_init__ after the inner object is constructed.

    # -- Collect class-level attributes to lift onto the wrapper -------------------
    class_attrs = {}

    # TOPICFILES / TOPICFILE
    if hasattr(cls, 'TOPICFILES'):
        class_attrs['TOPICFILES'] = cls.TOPICFILES
    if hasattr(cls, 'TOPICFILE'):
        class_attrs['TOPICFILE'] = cls.TOPICFILE

    # VERSION
    if hasattr(cls, 'VERSION'):
        class_attrs['VERSION'] = cls.VERSION

    # CONFIG — ensure it inherits from Datablock.CONFIG so that the
    # LazyLoader / specline mechanism works transparently.
    if hasattr(cls, 'CONFIG') and is_dataclass(cls.CONFIG):
        user_config = cls.CONFIG
        if issubclass(user_config, Datablock.CONFIG):
            class_attrs['CONFIG'] = user_config
        else:
            # Dynamically create a new dataclass that inherits from
            # Datablock.CONFIG with the same fields as the user's CONFIG.
            from dataclasses import MISSING, field as _dc_field_
            ns = {'__annotations__': {}}
            for f in fields(user_config):
                ns['__annotations__'][f.name] = f.type
                if f.default is not MISSING:
                    ns[f.name] = f.default
                elif f.default_factory is not MISSING:
                    ns[f.name] = _dc_field_(default_factory=f.default_factory)
                # else: no default — omit from namespace, dataclass will require it
            new_config = dataclass(
                type(user_config.__name__, (Datablock.CONFIG,), ns)
            )
            class_attrs['CONFIG'] = new_config
    else:
        # No CONFIG or not a dataclass — synthesize an empty one
        @dataclass
        class _EmptyCONFIG(Datablock.CONFIG):
            pass
        class_attrs['CONFIG'] = _EmptyCONFIG

    # -- If the Datablockable implements path(), lift it + block dirpath --------
    # With MI order (Datablock, cls), Datablock.path() would shadow the user's
    # path() since Datablock comes first in MRO.  We must explicitly lift it
    # into class_attrs so it takes priority.
    if hasattr(cls, 'path') and callable(getattr(cls, 'path')):
        def path(self, topic=None, *, ensure_dirpath=False):
            return cls.path(self, topic, ensure_dirpath=ensure_dirpath)
        class_attrs['path'] = path

        def dirpath(self, topic=None, *, ensure=False, list=False):
            raise NotImplementedError(
                f"{cls.__name__} defines its own path(); "
                f"the standard Datablock.dirpath() (root/anchor/hash/topic) "
                f"is not used. Use .path() instead."
            )
        class_attrs['dirpath'] = dirpath

    # -- Delegating methods -------------------------------------------------------
    def __post_init__(self):
        # The Datablockable's __init__ is NOT called when wrapped.
        # Datablock.__init__ already provides everything the wrapper needs:
        # cfg (cached_property), verbose/detailed/debug (properties),
        # log, TOPICFILE(S) (class-level, lifted into class_attrs).
        #
        # The Datablockable's __init__ is only for standalone (unwrapped) use.
        pass

    def __build__(self, *args, **kwargs):
        cls.__build__(self, *args, **kwargs)
        return self

    def __read__(self, topic=None):
        return cls.__read__(self, topic)

    def __reduce__(self):
        # Reconstruct via datablock(wrapped_cls) + __setstate__,
        # preserving the original __module__ so the anchor stays the same.
        return (_unpickle_datablock_instance, (cls, self.__class__.__module__, self.__getstate__()))

    @classmethod
    def from_datablockable(wrapper_cls, obj, *, root=None, **kwargs):
        """Create a Datablock wrapper from an existing datablockable instance.

        Extracts the CONFIG fields from *obj*.cfg as the ``spec`` dict,
        and propagates ``verbose``, ``detailed``, ``debug``
        from the datablockable instance to the wrapper.

        Parameters
        ----------
        obj : instance of the wrapped datablockable class
            Must have a ``.cfg`` attribute whose fields become the spec.
        root : str, optional
            Datablock root path.  If not given, falls back to DBX_ROOT.
        **kwargs
            Additional keyword arguments forwarded to the wrapper's
            ``__init__`` (e.g. ``tag``, ``anchored``, ``revision``).

        Returns
        -------
        Datablock
            A new wrapper instance ready for ``.build()`` / ``.read()``.

        Example
        -------
        ::

            @dbx.datablock
            class MyProcessor:
                ...

            proc = MyProcessor(paths=..., cfg=cfg, verbose=True, ...)
            block = MyProcessor.from_datablockable(proc, root='/data')
        """
        if not isinstance(obj, cls):
            raise TypeError(
                f"Expected an instance of {cls.__name__}, "
                f"got {type(obj).__name__}"
            )

        # Extract spec from the datablockable's CONFIG fields
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

        # Propagate observability / device settings from the datablockable
        for attr in ('verbose', 'detailed', 'debug', 'tag'):
            if attr not in kwargs and hasattr(obj, attr):
                kwargs[attr] = getattr(obj, attr)

        return wrapper_cls(root=root, spec=spec, **kwargs)

    class_attrs['__post_init__'] = __post_init__
    class_attrs['__build__'] = __build__
    class_attrs['__read__'] = __read__
    class_attrs['__reduce__'] = __reduce__
    class_attrs['from_datablockable'] = from_datablockable

    # -- Create the subclass dynamically ------------------------------------------
    wrapper_name = f'{cls.__name__}_Datablock'

    caller_module = inspect.currentframe().f_back.f_globals.get(
        '__name__', cls.__module__
    )

    WrapperClass = type(wrapper_name, (Datablock, cls), class_attrs)
    WrapperClass.__module__ = caller_module
    WrapperClass.__qualname__ = wrapper_name
    WrapperClass.__wrapped__ = cls

    return WrapperClass


def _unpickle_datablock_instance(cls, module, state):
    """Reconstruct a ``datablock(cls)`` wrapper instance from pickled state."""
    WrapperClass = datablock(cls)
    WrapperClass.__module__ = module
    obj = WrapperClass.__new__(WrapperClass)
    obj.__setstate__(state)
    return obj


def datastack(cls):
    """Wrap a Datastackable class as a :class:`Datastack` subclass.

    A class is *Datastackable* if it defines:

        SHARD = SomeDatablockableClass

        @dataclass
        class CONFIG:
            ...

        def __init__(self, *, cfg, verbose, detailed, debug, log):
            ...

        @property
        def n_shards(self) -> int:
            # Return the number of shards
            ...

        def __shard__(self, idx: int):
            # Return the SHARD instance for the given index
            ...

    Optional methods:

        def __read__(self, topic):  # if the stack produces its own output
            ...

    The wrapper creates a ``Datastack`` subclass that:

    1. Creates the inner Datastackable object (like ``datablock()``).
    2. Overrides ``n_shards`` and ``__shard__(idx)`` to delegate to the
       inner object, converting each returned Datablockable instance into
       a proper ``Datablock`` via ``from_datablockable()``.
    3. Inherits ``shard()``, ``shards()``, and ``__build__()`` from
       ``Datastack``, which builds shards in parallel using the selected
       ``DatablocksBuilder``.

    Usage::

        @dbx.datastack
        class MyPipeline:
            SHARD = MyProcessor   # a Datablockable class

            @dataclass
            class CONFIG:
                input_dir: str = None
                shard_size: int = 100

            def __init__(self, *, cfg, verbose, detailed, debug, log):
                self.cfg = cfg
                ...

            @property
            def n_shards(self):
                return compute_n_shards()

            def __shard__(self, idx):
                return MyProcessor(cfg=..., ...)

        stack = MyPipeline(root='/data', spec={...},
                           parallelization='multithreading', n_workers=4)
        stack.build()

    Parameters
    ----------
    cls : type
        A class satisfying the Datastackable protocol.

    Returns
    -------
    type
        A dynamically-created ``Datastack`` subclass wrapping *cls*.
    """
    from .datablocks import Datastack

    # -- Validate protocol --------------------------------------------------------
    if not hasattr(cls, '__shard__'):
        raise TypeError(f"{cls.__name__} must define __shard__ to be Datastackable")
    if not hasattr(cls, 'n_shards'):
        raise TypeError(f"{cls.__name__} must define n_shards to be Datastackable")
    if not hasattr(cls, 'SHARD'):
        raise TypeError(f"{cls.__name__} must define SHARD to be Datastackable")

    # -- Create the Datablock wrapper for this stack's shard class ----------------
    ShardBlock = datablock(cls.SHARD)

    # -- Collect class-level attributes -------------------------------------------
    class_attrs = {}

    # Store the shard block class for use by shards()
    class_attrs['_ShardBlock_'] = ShardBlock

    # TOPICFILES / TOPICFILE
    if hasattr(cls, 'TOPICFILES'):
        class_attrs['TOPICFILES'] = cls.TOPICFILES
    if hasattr(cls, 'TOPICFILE'):
        class_attrs['TOPICFILE'] = cls.TOPICFILE

    # VERSION
    if hasattr(cls, 'VERSION'):
        class_attrs['VERSION'] = cls.VERSION

    # CONFIG — ensure it inherits from Datablock.CONFIG
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

    # path() comes naturally via MRO from cls; no explicit delegation needed.

    # -- Delegating methods -------------------------------------------------------
    def __post_init__(self):
        # The Datastackable's __init__ is NOT called when wrapped.
        # Datablock.__init__ already provides everything the wrapper needs.
        # See datablock().__post_init__ for rationale.
        pass

    def n_shards_prop(self):
        return cls.n_shards.fget(self)
    class_attrs['n_shards'] = property(n_shards_prop)

    def __shard__(self, idx: int):
        """Convert a single Datastackable __shard__ output to a Datablock."""
        datablockable_shard = cls.__shard__(self, idx)
        return self._ShardBlock_.from_datablockable(datablockable_shard, root=self.root)

    def __reduce__(self):
        return (_unpickle_datastack_instance, (cls, self.__class__.__module__, self.__getstate__()))

    class_attrs['__post_init__'] = __post_init__
    class_attrs['__shard__'] = __shard__
    class_attrs['__reduce__'] = __reduce__

    # Optional __read__ delegation
    if hasattr(cls, '__read__'):
        def __read__(self, topic=None):
            return cls.__read__(self, topic)
        class_attrs['__read__'] = __read__

    # Optional __stack__ delegation
    if hasattr(cls, '__stack__'):
        def __stack__(self):
            return cls.__stack__(self)
        class_attrs['__stack__'] = __stack__

    # -- from_datastackable classmethod -------------------------------------------
    @classmethod
    def from_datastackable(wrapper_cls, obj, *, root=None, **kwargs):
        """Create a Datastack wrapper from an existing Datastackable instance.

        Extracts the CONFIG fields from *obj*.cfg as the ``spec`` dict,
        and propagates ``verbose``, ``detailed``, ``debug``.

        Parameters
        ----------
        obj : instance of the wrapped Datastackable class
        root : str, optional
        **kwargs : forwarded to wrapper ``__init__``

        Returns
        -------
        Datastack
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
        for attr in ('verbose', 'detailed', 'debug'):
            if attr not in kwargs and hasattr(obj, attr):
                kwargs[attr] = getattr(obj, attr)
        return wrapper_cls(root=root, spec=spec, **kwargs)

    class_attrs['from_datastackable'] = from_datastackable

    # -- Create the subclass dynamically ------------------------------------------
    wrapper_name = f'{cls.__name__}_Datastack'

    caller_module = inspect.currentframe().f_back.f_globals.get(
        '__name__', cls.__module__
    )

    WrapperClass = type(wrapper_name, (Datastack, cls), class_attrs)
    WrapperClass.__module__ = caller_module
    WrapperClass.__qualname__ = wrapper_name
    WrapperClass.__wrapped__ = cls

    # The ShardBlock was created by datablock() inside datastack(), so its
    # __module__ defaults to dbx.datawraps.  Fix it to match the caller.
    ShardBlock.__module__ = caller_module

    return WrapperClass


def _unpickle_datastack_instance(cls, module, state):
    """Reconstruct a ``datastack(cls)`` wrapper instance from pickled state."""
    WrapperClass = datastack(cls)
    WrapperClass.__module__ = module
    obj = WrapperClass.__new__(WrapperClass)
    obj.__setstate__(state)
    return obj

