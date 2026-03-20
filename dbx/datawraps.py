import inspect
from dataclasses import dataclass, fields, is_dataclass

from .dbx import Datablock


def datablock(cls):
    """Wrap a Datablockable class as a Datablock subclass.

    A class is *Datablockable* if it defines:

        TOPICFILES = {topic: filename, ...}   # or TOPICFILE = 'filename'
        VERSION = '...'                       # optional

        @dataclass
        class CONFIG:
            ...

        def __init__(self, *, paths, cfg, verbose, detailed, debug, log, device):
            ...

        def __build__(self, *args, **kwargs):
            ...
            return self

        def __read__(self, topic):
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
    if not (hasattr(cls, 'TOPICFILES') or hasattr(cls, 'TOPICFILE')):
        raise TypeError(
            f"{cls.__name__} must define TOPICFILES or TOPICFILE to be Datablockable"
        )

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

    # -- Helper: construct resolved paths for the inner object --------------------
    def _make_paths(wrapper_self):
        if wrapper_self.has_topics():
            return {topic: wrapper_self.path(topic) for topic in wrapper_self.topics()}
        else:
            return wrapper_self.path()

    # -- Delegating methods -------------------------------------------------------
    def __post_init__(self):
        paths = _make_paths(self)
        self.obj = cls(
            paths=paths,
            cfg=self.cfg,
            verbose=self.verbose,
            detailed=self.detailed,
            debug=self.debug,
            log=self.log,
            device=self.device,
        )

    def __build__(self, *args, **kwargs):
        self.obj.__build__(*args, **kwargs)
        return self

    def __read__(self, topic=None):
        return self.obj.__read__(topic)

    def __reduce__(self):
        # Reconstruct via datablock(wrapped_cls) + __setstate__,
        # preserving the original __module__ so the anchor stays the same.
        return (_unpickle_datablock_instance, (cls, self.__class__.__module__, self.__getstate__()))

    class_attrs['__post_init__'] = __post_init__
    class_attrs['__build__'] = __build__
    class_attrs['__read__'] = __read__
    class_attrs['__reduce__'] = __reduce__

    # -- Create the subclass dynamically ------------------------------------------
    wrapper_name = f'_{cls.__name__}_Datablock_'

    caller_module = inspect.currentframe().f_back.f_globals.get(
        '__name__', cls.__module__
    )

    WrapperClass = type(wrapper_name, (Datablock,), class_attrs)
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
