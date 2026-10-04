"""dbx.featuretables — Datablock / Datastack feature tables and bipolar encodings."""

from __future__ import annotations

from dataclasses import dataclass, field
import functools
import gc
import math
import warnings
from typing import Any

import numpy as np

# Suppress PyTorch non-writable NumPy array UserWarning for read-only streaming buffers
warnings.filterwarnings("ignore", category=UserWarning, message=".*given NumPy array is not writable.*")

try:
    import torch
except ImportError:
    torch = None

import dbx
from dbx.datablocks import DATADICT, DATADIR, DATAFILE, SAME, Datablock, Datastack, DIRTOPIC, forward_property
from dbx.backbones import ModelEvaluatorBuilder
from dbx.datatables import (
    DATASLICE,
    Datacollator,
    DataslicesUpstream,
    Datatab,
    Datatable,
    DatatableTab,
    DIRTOPIC,
    SLICETOPIC,
)
from dbx.datastreams import (
    ZipStreamingDataset,
    ZipIterableStreamingDatasets,
    column_spec,
    concat_data,
    merge_column_specs,
    project_column,
    slice_spec,
)


def _passthrough_collate_(batch):
    return batch


def _flatten_item_(x):
    while isinstance(x, dict) and len(x) > 0:
        x = next(iter(x.values()))
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32)
    if isinstance(x, np.ndarray):
        if x.dtype == np.object_:
            return np.array([_flatten_item_(item) for item in x], dtype=np.float32)
        return x.astype(np.float32) if x.dtype != np.float32 else x
    try:
        return np.asarray(x, dtype=np.float32)
    except Exception:
        return np.array(x)


def _to_tensor_(inputs, device=None) -> torch.Tensor:
    if isinstance(inputs, torch.Tensor):
        return inputs.to(device) if device is not None else inputs

    if isinstance(inputs, dict) and len(inputs) > 0:
        inputs = next(iter(inputs.values()))
        return _to_tensor_(inputs, device)

    if isinstance(inputs, np.ndarray):
        if inputs.dtype == np.object_:
            try:
                stacked = np.stack([_flatten_item_(x) for x in inputs])
                t = torch.from_numpy(np.ascontiguousarray(stacked))
            except Exception:
                stacked = np.array([_flatten_item_(x) for x in inputs], dtype=np.float32)
                t = torch.from_numpy(np.ascontiguousarray(stacked))
            return t.to(device) if device is not None else t
        t = torch.from_numpy(np.ascontiguousarray(inputs))
        return t.to(device) if device is not None else t

    if isinstance(inputs, (list, tuple)):
        if len(inputs) > 0 and isinstance(inputs[0], torch.Tensor):
            t = torch.stack(inputs)
            return t.to(device) if device is not None else t
        try:
            stacked = np.stack([_flatten_item_(x) for x in inputs])
            t = torch.from_numpy(np.ascontiguousarray(stacked))
        except Exception:
            stacked = np.array([_flatten_item_(x) for x in inputs], dtype=np.float32)
            t = torch.from_numpy(np.ascontiguousarray(stacked))
        return t.to(device) if device is not None else t

    try:
        t = torch.as_tensor(inputs)
    except Exception:
        t = torch.tensor(_flatten_item_(inputs))
    return t.to(device) if device is not None else t


def feature_map(feature_for_column, evaluator_factory) -> dict[str, str]:
    """``{column: layer}``: which evaluator layer each ``features`` column holds.

    See `Featuretab.VAR.feature_for_column`: a dict is the map itself; a list,
    tuple or single string keeps just those layers, each under its own name;
    None keeps every layer in ``evaluator_factory.layer_names``.
    """
    if feature_for_column is None:
        layer_names = evaluator_factory.layer_names if evaluator_factory is not None else []
        return {name: name for name in layer_names}
    if isinstance(feature_for_column, dict):
        return dict(feature_for_column)
    if isinstance(feature_for_column, (list, tuple)):
        return {str(k): str(k) for k in feature_for_column}
    return {str(feature_for_column): str(feature_for_column)}


class Featuretab(DataslicesUpstream, Datatab):
    """A tab storing multi-layer feature activations captured by an evaluator.

    Inherits access to the slices of its ``upstream`` tab, the one its features
    were computed from. Calling `dataset()` or `data()` with slice names present
    upstream zips them in using the `ZipStreamingDataset` mechanism.
    """

    VERSION = 1

    @forward_property({'features': DATASLICE})
    def TOPICS(self):
        """One ``ndarray:float32`` column per feature: the evaluator's layers, or `feature_for_column`'s columns.

        On the class -- what a table sees of its TAB -- the slice alone: the
        columns are each tab's spec's to say.
        """
        return {'features': DATASLICE({column: 'ndarray:float32' for column in self.feature_for_column})}

    SPECIALIZATIONS = [
        Datablock.Specialization(
            spec={}, topics=SAME, redirect_vars={'upstream': 'datapoint_tab'},
            note="VAR field renamed from datapoint_tab to upstream"),
        Datablock.Specialization(
            spec={}, topics=SAME, redirect_vars={'upstream': 'datapoint_tab', 'feature_for_column': 'feature_namemap'},
            note="... and from feature_namemap to feature_for_column"),
        Datablock.Specialization(
            spec={}, topics={'features': SLICETOPIC},
            redirect_vars={'upstream': 'datapoint_tab', 'feature_for_column': 'feature_namemap'},
            note="respelled: the sentinel for DATASLICE, and the field names before"),
    ]

    @dataclass
    class VAR(Datablock.VAR):
        upstream: Datatab
        evaluator_factory: ModelEvaluatorBuilder
        collator: Datacollator
        #: The feature -- the evaluator layer -- for each column of the
        #: `features` slice: ``{column: layer}``.  ``{"custom_output": "final"}``
        #: writes the evaluator's ``final`` layer as the column
        #: ``custom_output``.  None keeps every layer in
        #: ``evaluator_factory.layer_names``, each under its own name; a list,
        #: tuple or single string keeps just those layers, likewise unrenamed.
        #:
        #: The keys are the slice's declared columns.  A layer the evaluator
        #: does not return is skipped at build time without complaint, so its
        #: column is declared but never written.  Being VAR, it is part of the
        #: block's identity: tabs differing only here are different blocks.
        feature_for_column: dict[str, str] | None = None
        shard_size_limit_bytes: int = 1 << 26  # 64 MiB default, in bytes

    # 1. Protocol and hooks ------------------------------------------------

    # TODO: carry `shared_upstream_column` through into the features slice at
    # build time, so the cross-block alignment check has something to compare.
    #
    # As it stands the features slice declares `feature_for_column`'s columns and
    # nothing else. A stock feature tab
    # therefore has no column in common with its upstream sample slices, and
    # setting shared_upstream_column= on one is refused at read time ("carried
    # by fewer than two of the sources read") rather than checking anything.
    # Which is the right refusal -- it says the check cannot run -- but it means
    # the declaration is only usable by a subclass whose __build__ writes an id
    # of its own.
    #
    # Three things it needs, and why it was not done alongside the read-time
    # default:
    #
    #   * It must be VAR, not the kwarg it is now. Writing the column changes
    #     the bytes, so two feature tabs -- one carrying it, one not -- hold
    #     different data, and as a kwarg they would share a hash. That is the
    #     fault VAR.LazyLoader._check_renderable_ exists to prevent, arrived at
    #     from the other side.
    #   * It needs the column's MDS type, which only a DATASLICE-declared
    #     upstream states (`upstream.declared_columns(slice)`). Under the
    #     SLICETOPIC sentinel nothing declares it, and inferring a type from a
    #     value that has already been through MDS is guesswork -- so this may
    #     have to require a declared upstream, and say so.
    #   * Both build paths have to write it: __build_bulk__ can take it out of
    #     the `sample_data` it already read, but __build_streaming__ sees rows
    #     only as whatever _passthrough_collate_ made of the batch.
    def __init__(
        self,
        *args,
        device_batch_size: int = 64,
        device: str = "cpu",
        streaming: bool = False,
        dataloader_kwargs: dict | None = None,
        shared_upstream_column: 'str | None' = None,
        **kwargs,
    ):
        super().__init__(
            *args,
            device_batch_size=device_batch_size,
            device=device,
            streaming=streaming,
            dataloader_kwargs=dataloader_kwargs,
            shared_upstream_column=shared_upstream_column,
            **kwargs,
        )

    def __post_init__(self):
        super().__post_init__()
        self.device = getattr(self, 'device', 'cpu')
        self.device_batch_size = getattr(self, 'device_batch_size', 64)
        self.streaming = getattr(self, 'streaming', False)
        self.dataloader_kwargs = getattr(self, 'dataloader_kwargs', None) or {}

    def __build__(self):
        if self.streaming:
            return self.__build_streaming__()
        else:
            return self.__build_bulk__()

    def __build_bulk__(self):
        evaluator = self.var.evaluator_factory.evaluator(device=self.device, log=self.log)
        upstream = self.var.upstream

        collator = self.var.collator
        with self.slice_writers(size_limit=self.var.shard_size_limit_bytes) as writers:
            sample_data = upstream.data(*collator.slices(upstream), concat=True)
            inputs = collator(sample_data, signal_only=True)
            inputs = _to_tensor_(inputs, "cpu")

            n_samples = len(inputs)
            n_batches = math.ceil(n_samples / self.device_batch_size)

            for k in range(n_batches):
                m = k * self.device_batch_size
                n = min((k + 1) * self.device_batch_size, n_samples)
                batch = inputs[m:n].to(self.device)
                result = evaluator(batch)

                batch_len = n - m
                batch_features = {
                    col_name: result[layer_name].cpu().numpy().astype(np.float32)
                    for col_name, layer_name in self.feature_for_column.items()
                    if layer_name in result
                }
                for i in range(batch_len):
                    writers['features'].write({col_name: arr[i] for col_name, arr in batch_features.items()})
                # Freed by reference count, and reused by the CUDA caching allocator
                # for the next batch of the same shape: no collection per batch.
                del batch, result, batch_features
            del inputs
        # Once per bag: collect, and hand the allocator's cache back -- see `__build_streaming__`.
        evaluator.clear()
        del evaluator
        gc.collect()
        return self

    def __build_streaming__(self):
        warnings.filterwarnings("ignore", category=UserWarning, message=".*given NumPy array is not writable.*")
        gc.collect()
        evaluator = self.var.evaluator_factory.evaluator(device=self.device, log=self.log)
        upstream = self.var.upstream

        collator = self.var.collator

        dataset = upstream.dataset(*collator.slices(upstream))
        dl_kwargs = dict(self.dataloader_kwargs) if self.dataloader_kwargs else {}
        dl_kwargs.setdefault('batch_size', self.device_batch_size)
        dl_kwargs.setdefault('collate_fn', _passthrough_collate_)


        dataloader = torch.utils.data.DataLoader(dataset, **dl_kwargs)

        with self.slice_writers(size_limit=self.var.shard_size_limit_bytes) as writers:
            for batch_data in dataloader:
                inputs = collator(batch_data, signal_only=True)
                batch = _to_tensor_(inputs, self.device)
                result = evaluator(batch)

                batch_len = len(batch)
                batch_features = {
                    col_name: result[layer_name].cpu().numpy().astype(np.float32)
                    for col_name, layer_name in self.feature_for_column.items()
                    if layer_name in result
                }
                for i in range(batch_len):
                    writers['features'].write({col_name: arr[i] for col_name, arr in batch_features.items()})
                # Freed by reference count, and reused by the CUDA caching allocator for
                # the next batch of the same shape. A gc.collect() and
                # torch.cuda.empty_cache() here -- evaluator.clear() -- ran per batch,
                # stalling the GPU worker every batch for no memory it did not get back anyway.
                del batch_data, inputs, batch, result, batch_features
        # Once per bag: collect, and hand the allocator's cache back to the driver,
        # so a worker between bags holds no more than it needs.
        evaluator.clear()
        del dataloader, dataset, evaluator
        gc.collect()
        return self

    def __len__(self) -> int:
        return len(self.var.upstream)

    # 3. Accessors ---------------------------------------------------------

    @property
    def feature_columns(self) -> list[str]:
        """The columns of the ``features`` slice, in declared order."""
        return list(self.feature_for_column)

    @functools.cached_property
    def feature_for_column(self) -> dict[str, str]:
        """``{column: layer}``, from `VAR.feature_for_column` and the evaluator's layers."""
        return feature_map(self.var.feature_for_column, self.var.evaluator_factory)

    # 4. Helpers -----------------------------------------------------------


class Featuretable(DataslicesUpstream, Datatable):
    """A table of `Featuretab` blocks built across a `Datatable`."""

    TAB = Featuretab
    VERSION = 1

    SPECIALIZATIONS = [
        Datablock.Specialization(
            spec={}, topics=SAME, redirect_vars={'upstream': 'datapoint_table'},
            note="VAR field renamed from datapoint_table to upstream"),
        Datablock.Specialization(
            spec={}, topics=SAME, redirect_vars={'upstream': 'datapoint_table', 'feature_for_column': 'feature_namemap'},
            note="... and from feature_namemap to feature_for_column"),
    ]

    @dataclass
    class VAR(Datablock.VAR):
        upstream: Datatable
        evaluator_factory: ModelEvaluatorBuilder
        collator: Datacollator
        #: Passed unchanged to every tab; see `Featuretab.VAR.feature_for_column`.
        feature_for_column: dict | None = None
        shard_size_limit_bytes: int = 1 << 26  # 64 MiB default, in bytes

    # 1. Protocol and hooks ------------------------------------------------

    def __init__(
        self,
        *args,
        device_batch_size: int = 64,
        devices: list | None = None,
        device_parallelization: str | None = None,
        device_work_stealing: bool = False,
        streaming: bool = False,
        dataloader_kwargs: dict | None = None,
        filter_built_tabs: bool = False,
        shared_upstream_column: 'str | None' = None,
        **kwargs,
    ):
        # The feature evaluation -- building the tabs -- queues off *devices*,
        # one worker per device: *device_parallelization* (default:
        # multiprocessing for several devices, inline for one) and
        # *device_work_stealing*. Everything else -- checking and adopting the
        # tabs -- runs at the stack's own *n_workers*, *parallelization* and
        # *work_stealing*. Operational, and passed on only when set.
        if device_parallelization is not None:
            kwargs['device_parallelization'] = device_parallelization
        if device_work_stealing:
            kwargs['device_work_stealing'] = True
        super().__init__(
            *args,
            device_batch_size=device_batch_size,
            devices=devices,
            streaming=streaming,
            dataloader_kwargs=dataloader_kwargs,
            filter_built_tabs=filter_built_tabs,
            shared_upstream_column=shared_upstream_column,
            **kwargs,
        )

    def __post_init__(self):
        super().__post_init__()
        # Read back from self — works whether we came through __init__ or __setstate__.
        self.streaming = getattr(self, 'streaming', False)
        self.dataloader_kwargs = getattr(self, 'dataloader_kwargs', None) or {}
        self.filter_built_tabs = getattr(self, 'filter_built_tabs', False)
        self.device_batch_size = getattr(self, 'device_batch_size', 64)
        self.devices = getattr(self, 'devices', None) or ["cpu"]
        self._devices = self.devices

    def build(self, *args, **kwargs):
        """As `Datatable.build`, after checking that its *devices* exist here -- before anything else is done."""
        self._check_devices_()
        return super().build(*args, **kwargs)

    def _build_executor_(self, tag: str):
        """The feature evaluation's executor: one worker per device -- see `__init__`."""
        devices = list(self.devices)
        key = (getattr(self, 'device_parallelization', None)
               or ('multiprocessing' if len(devices) > 1 else 'inline')).lower()
        executors = self._get_executors_()
        if key not in executors:
            raise ValueError(f"Unknown device_parallelization {key!r}. Choose from {list(executors)}")
        cls = executors[key]
        kwargs = self._executor_kwargs_(tag=f"{tag} on {devices}", n_workers=len(devices), executor_cls=cls)
        kwargs.pop('work_stealing', None)
        if getattr(self, 'device_work_stealing', False):
            kwargs['work_stealing'] = True
        self.log.info(f"{self.anchorkeypath}: evaluating features on {devices}, one worker per device ({key})")
        return cls(**kwargs)

    def _check_devices_(self) -> None:
        """Raise for a ``cuda:<i>`` this process cannot see.

        Each worker takes one of *devices*; one naming a GPU the job was not
        given fails as ``CUDA error: invalid device ordinal`` -- in a worker,
        after the blocks are checked and the journal read, minutes in. Here it
        fails first, saying how many devices there are.
        """
        cuda = [str(d) for d in self.devices if str(d).startswith('cuda')]
        if not cuda:
            return
        if torch is None or not torch.cuda.is_available():
            raise RuntimeError(f"{type(self).__name__}: devices={self.devices} asks for CUDA, and this process "
                               f"has none (torch.cuda.is_available() is False)")
        n = torch.cuda.device_count()
        missing = [d for d in cuda if (int(d.split(':', 1)[1]) if ':' in d else 0) >= n]
        if missing:
            raise ValueError(f"{type(self).__name__}: devices={self.devices} names {missing}, but this process "
                             f"sees {n} CUDA device(s), cuda:0..cuda:{n - 1} -- ask for no more than the job has")

    def __tab__(self, idx: int, device: str | None = None, tag=None) -> Featuretab:
        upstream = self.var.upstream.tab(idx)
        spec = dict(
            upstream=upstream.quote(),
            evaluator_factory=self.spec['evaluator_factory'],
            collator=self.spec.get('collator'),
            feature_for_column=self.spec.get('feature_for_column'),
            shard_size_limit_bytes=self.spec.get('shard_size_limit_bytes', 1 << 26),
        )
        tab_specs = (getattr(self, 'TAB_SPECIALIZATIONS', None) or getattr(self, 'TAB_SPECIALIZATION', None)
                     or getattr(self, 'BLOCK_SPECIALIZATIONS', None) or getattr(self, 'BLOCK_SPECIALIZATION', None))
        return self.TAB(
            datalake=self._datalake_,
            storage_options=self.storage_options,
            cache=getattr(self, 'cache', None),
            cache_limit=getattr(self, 'cache_limit', None),
            verbose=False,
            spec=spec,
            SPECIALIZATIONS=tab_specs,
            device_batch_size=self.device_batch_size,
            device=device,
            streaming=self.streaming,
            dataloader_kwargs=self.dataloader_kwargs,
            revision=self.revision,
            tag=tag if tag is not None else upstream.tag,
        )

    def __block__(self, idx: int, **kwargs) -> Featuretab:
        return self.__tab__(idx, **kwargs)

    # 2. Declared API ------------------------------------------------------

    def validate_tab(self, i: int, **kwargs) -> bool:
        """Return whether the tab at index *i* validates."""
        signature1 = self.tab(i).var.upstream.signaturestr()
        signature2 = self.var.upstream.tab(i).signaturestr()
        coherent_signatures = (signature1 == signature2)
        if not coherent_signatures:
            raise ValueError(f"tab({i}).var.upstream.signature: {signature1} "
                             f"does not match var.upstream.tab({i}): {signature2}")
        feature_dataset_len = self.tab(i).dataset().__len__()
        point_dataset_len = self.tab(i).var.upstream.dataset().__len__()
        coherent_datasets = feature_dataset_len == point_dataset_len
        if not coherent_datasets:
            raise ValueError(f"tab({i}).dataset().__len__(): {feature_dataset_len} does not match"
                             f"tab({i}).var.upstream.dataset().__len__(): {point_dataset_len}"
            )
        return coherent_signatures and coherent_datasets and self.tab(i).validate(**kwargs)

    # 3. Accessors ---------------------------------------------------------

    @property
    def feature_columns(self) -> list[str]:
        """The columns of every tab's ``features`` slice, in declared order."""
        return list(self.feature_for_column)

    @functools.cached_property
    def feature_for_column(self) -> dict[str, str]:
        """``{column: layer}``, as each of its tabs has it."""
        return feature_map(self.var.feature_for_column, self.var.evaluator_factory)

    @property
    def n_tabs(self) -> int:
        return self.var.upstream.n_tabs


class BipolarFeaturetab(DataslicesUpstream, Datatab):
    """Bipolar encoding of a `Featuretab`'s feature columns against a calibration.

    Each column is mapped to ``{-1, +1}`` elementwise: ``+1`` where the value
    is at or above the column's median, ``-1`` below it. The median is the
    whole-table one in ``VAR.stats_probe`` -- a `FeatureStatsProbe` over a
    calibration table, usually another fold than the one this tab is in -- and
    the values are normalized first exactly as that probe normalized them
    before taking it (`normalize_features` with its ``normalization``, one
    column at a time), so value and median are in the same space.

    Not the tab's own median. Centring every tab on itself makes each column
    exactly half ``+1`` within every tab, which erases whatever sets one tab
    apart from another -- for slides, the very signal a slide-level probe is
    after -- and the encoding then describes only variation within a tab.

    One ``bipolar`` column per encoded feature column, of the same name, so
    ``('bipolar', c)`` is the encoding of ``('features', c)``. Every column of
    the featuretab's ``features`` slice by default; ``VAR.features`` names a
    subset. The columns are declared per instance, from the featuretab's own
    declaration.
    """

    VERSION = 2

    SPECIALIZATIONS = [
        Datablock.Specialization(
            spec={}, topics=SAME, redirect_vars={'upstream': 'featuretab'},
            note="VAR field renamed from featuretab to upstream"),
    ]

    @forward_property({'bipolar': DATASLICE})
    def TOPICS(self):
        """One ``ndarray:int8`` column per encoded feature column, of the same name.

        On the class -- what a table sees of its TAB -- the slice alone.
        """
        return {'bipolar': DATASLICE(**{c: 'ndarray:int8' for c in self.feature_columns})}

    @dataclass
    class VAR(Datablock.VAR):
        upstream: Featuretab
        stats_probe: 'FeatureStatsProbe'
        features: list | None = None
        datapoints_per_row: int = 1

    # 1. Protocol and hooks ------------------------------------------------

    def __post_init__(self):
        super().__post_init__()
        self._check_stats_probe_()

    def __build__(self):
        probe = self.stats_probe
        mode = probe.var.normalization
        columns = self.feature_columns
        data = self.upstream.data(('features', list(columns)), concat=True)['features']

        from dbx.probes import normalize_features
        bipolar, counts = {}, set()
        for c in columns:
            x = np.asarray(data[c]).astype(np.float64)
            if mode is not None:
                x = np.asarray(normalize_features(x, mode))
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            if np.isnan(x).any():
                raise ValueError(f"{type(self).__name__}: column 'features.{c}' of {self.upstream.tag!r} "
                                 f"holds NaN, which is neither above nor below a median")
            median = np.asarray(probe.stat('median', ('features', c)))
            if median.shape != x.shape[1:]:
                raise ValueError(
                    f"{type(self).__name__}: column 'features.{c}' has samples of shape "
                    f"{x.shape[1:]}, but the median in {probe.anchorkeypath} has shape "
                    f"{median.shape}: the probe describes a different feature."
                )
            bipolar[c] = np.where(x >= median, 1, -1).astype(np.int8)
            counts.add(len(x))
        del data

        if len(counts) != 1:
            raise ValueError(f"{type(self).__name__}: feature columns disagree on row count: {sorted(counts)}")
        with self.slice_writers() as writers:
            for i in range(counts.pop()):
                writers['bipolar'].write({c: bipolar[c][i] for c in columns})
        return self

    def __len__(self) -> int:
        return len(self.upstream)

    # 3. Accessors ---------------------------------------------------------

    @property
    def upstream(self) -> Featuretab:
        return self.var.upstream

    @property
    def stats_probe(self):
        return self.var.stats_probe

    @property
    def feature_columns(self) -> list[str]:
        """The featuretab columns encoded here, in the featuretab's declared order."""
        declared = self.upstream.declared_columns('features')
        if not declared:
            raise ValueError(
                f"{type(self).__name__}: {self.upstream.anchorkeypath} declares no columns "
                f"for its 'features' slice, so there is nothing to say which to encode. "
                f"Encode a featuretab whose slice is a declared DATASLICE."
            )
        if self.var.features is None:
            return list(declared)
        unknown = [c for c in self.var.features if c not in declared]
        if unknown:
            raise ValueError(f"{type(self).__name__}: features {unknown} are not columns of "
                             f"the featuretab's 'features' slice {list(declared)}")
        return [c for c in declared if c in self.var.features]

    # 4. Helpers -----------------------------------------------------------

    def _check_stats_probe_(self) -> None:
        """Refuse, at construction, a stats probe that cannot calibrate these columns."""
        from dbx.probes import FeatureStatsProbe
        probe = self.stats_probe
        if not isinstance(probe, FeatureStatsProbe):
            raise TypeError(f"{type(self).__name__}: stats_probe must be a FeatureStatsProbe, "
                            f"got {type(probe).__name__}")
        missing = [c for c in self.feature_columns if ('features', c) not in probe.column_paths]
        if missing:
            raise ValueError(
                f"{type(self).__name__}: stats_probe {probe.anchorkeypath} has no median for "
                f"feature column(s) {missing}; it describes {probe.column_paths}. Give it a "
                f"collator whose signals include {[('features', c) for c in missing]}."
            )


class BipolarFeaturetable(DataslicesUpstream, Datatable):
    """A table of `BipolarFeaturetab` blocks built over a `Featuretable`.

    Every tab is calibrated by the one ``stats_probe``: that is what makes the
    tabs' encodings comparable with each other.
    """

    TAB = BipolarFeaturetab
    VERSION = 2

    SPECIALIZATIONS = [
        Datablock.Specialization(
            spec={}, topics=SAME, redirect_vars={'upstream': 'featuretable'},
            note="VAR field renamed from featuretable to upstream"),
    ]

    @dataclass
    class VAR(Datatable.VAR):
        upstream: Featuretable = None
        stats_probe: 'FeatureStatsProbe' = None
        features: list | None = None

    # 1. Protocol and hooks ------------------------------------------------

    def __post_init__(self):
        super().__post_init__()
        if self.var.upstream is None or self.var.stats_probe is None:
            raise ValueError(f"{type(self).__name__}: VAR.upstream and VAR.stats_probe are both required")

    def __tab__(self, idx: int, tag=None, **kwargs) -> BipolarFeaturetab:
        upstream = self.var.upstream.tab(idx)
        tab_specs = (getattr(self, 'TAB_SPECIALIZATIONS', None) or getattr(self, 'TAB_SPECIALIZATION', None)
                     or getattr(self, 'BLOCK_SPECIALIZATIONS', None) or getattr(self, 'BLOCK_SPECIALIZATION', None))
        return self.TAB(
            datalake=self._datalake_,
            storage_options=self.storage_options,
            cache=getattr(self, 'cache', None),
            cache_limit=getattr(self, 'cache_limit', None),
            verbose=False,
            spec=dict(
                upstream=dbx.quote(upstream),
                stats_probe=dbx.quote(self.var.stats_probe),
                features=self.var.features,
            ),
            SPECIALIZATIONS=tab_specs,
            revision=self.revision,
            tag=tag if tag is not None else upstream.tag,
        )

    def __block__(self, idx: int, **kwargs) -> BipolarFeaturetab:
        return self.__tab__(idx, **kwargs)

    # 2. Accessors ---------------------------------------------------------

    @property
    def upstream(self) -> Featuretable:
        return self.var.upstream

    @property
    def stats_probe(self):
        return self.var.stats_probe

    @property
    def n_tabs(self) -> int:
        return self.upstream.n_tabs


class TernaryFeaturetab(DataslicesUpstream, Datatab):
    """A `BipolarFeaturetab`'s tiles, kept where their bag is sure of the sign and they agree with it, else 0.

    The bag-level estimate, per column and dimension, is the mean of the
    tab's bipolar rows, ``m = 2p - 1`` where ``p`` is the fraction at ``+1``:
    how far the bag leans, and so how sure its sign is. The bag's sign is
    ``sign(m)`` where ``|m| >= bag_threshold``, and 0 -- undecided -- where it
    leans less. A row's ternary value is its bipolar value where the bag's
    sign is decided and the row agrees with it, and 0 otherwise: ``{-1, 0, +1}``,
    0 marking a dimension where the tile does not speak for its bag.

    This is the June BitPath encoding (``bag_aggregation_threshold``, 0.5,
    with its tiles ternarized), now a block of its own: the targets it gives a
    still are an artifact with an identity, built once, rather than computed
    inside a dataset.

    Topics: ``ternary`` -- one ``ndarray:int8`` column per bipolar column, of
    the same name; ``bag/<column>`` -- ``mean``, ``sign`` and ``n_rows``, the
    estimate each row was judged against, from which another rule (an
    interval on ``p``, say) can be read without a rebuild.
    """

    VERSION = 1

    BAG_DATADICT = DATADICT('bag.npz', mean='ndarray', sign='ndarray', n_rows='ndarray')

    @forward_property({'ternary': DATASLICE})
    def TOPICS(self):
        """``ternary``, one ``ndarray:int8`` column per bipolar column, and ``bag/<column>``.

        On the class -- what a table sees of its TAB -- the slice alone.
        """
        columns = self.bipolar_columns
        return {'ternary': DATASLICE(**{c: 'ndarray:int8' for c in columns}),
                'bag': {c: self.BAG_DATADICT for c in columns}}

    @dataclass
    class VAR(Datablock.VAR):
        upstream: BipolarFeaturetab
        bag_threshold: float = 0.5

    # 1. Protocol and hooks ------------------------------------------------

    def __post_init__(self):
        super().__post_init__()
        if not 0 < float(self.var.bag_threshold) <= 1:
            raise ValueError(f"{type(self).__name__}: bag_threshold is how far a bag's mean bipolar value "
                             f"must lean to decide its sign, in (0, 1]; got {self.var.bag_threshold!r}")

    def __build__(self):
        from dbx.dataparts import write_npz
        columns = self.bipolar_columns
        data = self.upstream.data(('bipolar', list(columns)), concat=True)['bipolar']
        ternary, counts = {}, set()
        for c in columns:
            x = np.asarray(data[c]).astype(np.int8)
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            if not np.isin(x, (-1, 1)).all():
                raise ValueError(f"{type(self).__name__}: column 'bipolar.{c}' of {self.upstream.tag!r} "
                                 f"holds values other than -1 and +1: {np.unique(x)[:8].tolist()}")
            mean = x.astype(np.float64).mean(axis=0)
            sign = np.where(np.abs(mean) >= self.var.bag_threshold, np.sign(mean), 0).astype(np.int8)
            ternary[c] = np.where((sign != 0) & (x == sign), x, 0).astype(np.int8)
            write_npz(self.path('bag', c, ensure_dirpath=True), storage_options=self.storage_options,
                      mean=mean.astype(np.float32), sign=sign, n_rows=np.asarray(len(x)))
            counts.add(len(x))
        del data
        if len(counts) != 1:
            raise ValueError(f"{type(self).__name__}: bipolar columns disagree on row count: {sorted(counts)}")
        with self.slice_writers(['ternary']) as writers:
            for i in range(counts.pop()):
                writers['ternary'].write({c: ternary[c][i] for c in columns})
        return self

    def __len__(self) -> int:
        return len(self.upstream)

    # 2. Declared API ------------------------------------------------------

    def bag(self, column: str) -> dict:
        """``{'mean', 'sign', 'n_rows'}`` of *column*: the bag-level estimate its rows were judged against."""
        from dbx.dataparts import read_npz
        if column not in self.bipolar_columns:
            raise KeyError(f"{type(self).__name__}: no bipolar column {column!r}; it has {self.bipolar_columns}")
        return read_npz(self.path('bag', column), 'mean', 'sign', 'n_rows', storage_options=self.storage_options)

    # 3. Accessors ---------------------------------------------------------

    @property
    def upstream(self) -> BipolarFeaturetab:
        return self.var.upstream

    @property
    def bipolar_columns(self) -> list[str]:
        """The upstream's bipolar columns, in its declared order."""
        declared = self.upstream.declared_columns('bipolar')
        if not declared:
            raise ValueError(f"{type(self).__name__}: {self.upstream.anchorkeypath} declares no columns for "
                             f"its 'bipolar' slice, so there is nothing to say which to ternarize")
        return list(declared)


class TernaryFeaturetable(DataslicesUpstream, Datatable):
    """A table of `TernaryFeaturetab` blocks built over a `BipolarFeaturetable`, one threshold for all."""

    TAB = TernaryFeaturetab
    VERSION = 1

    @dataclass
    class VAR(Datatable.VAR):
        upstream: BipolarFeaturetable = None
        bag_threshold: float = 0.5

    # 1. Protocol and hooks ------------------------------------------------

    def __post_init__(self):
        super().__post_init__()
        if self.var.upstream is None:
            raise ValueError(f"{type(self).__name__}: VAR.upstream is required")

    def __tab__(self, idx: int, tag=None, **kwargs) -> TernaryFeaturetab:
        upstream = self.var.upstream.tab(idx)
        tab_specs = (getattr(self, 'TAB_SPECIALIZATIONS', None) or getattr(self, 'TAB_SPECIALIZATION', None)
                     or getattr(self, 'BLOCK_SPECIALIZATIONS', None) or getattr(self, 'BLOCK_SPECIALIZATION', None))
        return self.TAB(
            datalake=self._datalake_,
            storage_options=self.storage_options,
            cache=getattr(self, 'cache', None),
            cache_limit=getattr(self, 'cache_limit', None),
            verbose=False,
            spec=dict(upstream=dbx.quote(upstream), bag_threshold=self.var.bag_threshold),
            SPECIALIZATIONS=tab_specs,
            revision=self.revision,
            tag=tag if tag is not None else upstream.tag,
        )

    def __block__(self, idx: int, **kwargs) -> TernaryFeaturetab:
        return self.__tab__(idx, **kwargs)

    # 3. Accessors ---------------------------------------------------------

    @property
    def upstream(self) -> BipolarFeaturetable:
        return self.var.upstream

    @property
    def n_tabs(self) -> int:
        return self.upstream.n_tabs
