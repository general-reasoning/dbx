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
from dbx.datablocks import DATADIR, DATAFILE, SAME, Datablock, Datastack, DIRTOPIC, forward_property
from dbx.backbones import ModelEvaluatorBuilder
from dbx.datatables import (
    DATASLICE,
    DataslicesUpstream,
    Datatab,
    Datatable,
    DatapointTableTab,
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


def _extract_pair_data_(data_dict, pair: tuple[str, str]):
    s_name, c_name = pair[0], pair[1]
    if isinstance(data_dict, dict) and s_name in data_dict:
        val = data_dict[s_name]
        if isinstance(val, dict):
            if c_name in val:
                return val[c_name]
            col_key = c_name.replace('features_', '')
            if col_key in val:
                return val[col_key]
            return next(iter(val.values()))
        return val
    return data_dict


class Datacollator(Datablock):
    """Callable Datablock for collating batches of datapoint dicts into signal and label arrays.

    `Datacollator` has no `TOPICS` (it does not build or persist files).
    When invoked as `collator(datapoints)`, it extracts the specified `signals` and `labels`
    `(slice, column)` pairs from each datapoint dict in `datapoints`, concatenating/stacking
    signal tensors along a new dimension 1 for each datapoint, and concatenating datapoints
    along dimension 0 (batch dimension).

    ``labels`` defaults to None: a collator for signals alone -- a feature build,
    an unsupervised pass -- names no label slice, reads none, and returns
    ``(signals,)``.

    A pair may go deeper, into a column that holds a dict, read as `dataset()`
    reads a request -- TUPLES are depth, LISTS are several side by side:
    ``('annotations', 'annotations', 'label')`` is ``value['label']``,
    ``('annotations', 'annotations', 'site', 'code')`` -- or
    ``(..., ('site', 'code'))`` -- is ``value['site']['code']``, and a list of
    keys, paths or columns is one pair per item. The entry is taken where the
    value is picked, so it works on a row and on a stacked batch alike.
    """

    TOPICS = {}

    @dataclass
    class VAR(Datablock.VAR):
        signals: list[tuple[str, str]]
        labels: list[tuple[str, str]] | None = None
        length: int | None = None
        skip_missing: bool = False

    # 1. Protocol and hooks ------------------------------------------------

    def __call__(self, datapoints, *, signal_only: bool = False):
        """Collate a batch of datapoint rows into signal (and label) arrays.

        Parameters
        ----------
        datapoints : list[dict] | dict
            Either one row per sample -- what a `DataLoader` over
            `dataset()` yields -- or one already-stacked mapping per slice,
            which is what `data(*collator.slices(), concat=True)` hands back.
            Both are keyed ``{slice: {column: ...}}``; they are told apart by
            whether the outer object is a list, not by a flag, because the
            callers that pass a whole slice at a time (a feature build, a
            probe fit) are the same ones that pass batches elsewhere.
        signal_only : bool
            Return the signal array bare, rather than a tuple.

        Returns
        -------
        np.ndarray | tuple[np.ndarray, ...]
            The signal array alone when *signal_only*; otherwise
            ``(signals, labels)``, or ``(signals,)`` when no labels are
            declared. Never a dict: every caller either unpacks positionally
            or wants the one array, and a mapping only invited the two to be
            addressed by names that had to agree across three files.
        """
        sig_arr = self._collate_pairs_(datapoints, self.var.signals)

        length = self.var.length
        if length is not None and getattr(sig_arr, 'ndim', 0) >= 1 and sig_arr.shape[-1] > length:
            sig_arr = sig_arr[..., :length]

        if signal_only:
            return sig_arr

        if not self.var.labels:
            return (sig_arr,)

        lbl_arr = self._collate_pairs_(datapoints, self.var.labels)
        if length is not None and getattr(lbl_arr, 'ndim', 0) >= 1 and lbl_arr.shape[-1] > length:
            lbl_arr = lbl_arr[..., :length]
        return (sig_arr, lbl_arr)

    # 2. Declared API ------------------------------------------------------

    def slices(self):
        """The slices these pairs name, deduplicated, in declaration order.

        Order-preserving rather than ``set``-derived: this is splatted into
        ``dataset(*collator.slices())`` and ``data(*collator.slices())``, where
        position decides the order sources are zipped in. Python hashes str
        with a per-process seed, so a set here put a different slice order in
        front of every worker and every rerun -- which is not something a
        config-addressed build can afford, and not something it would report.
        """
        seen = {}
        for pair in self.signal_pairs + self.label_pairs:
            seen[pair[0]] = None
        return list(seen)

    # 3. Accessors ---------------------------------------------------------

    @property
    def signal_pairs(self) -> tuple[tuple[str, ...], ...]:
        """The signal ``(slice, column)`` pairs, each in full two-part form.

        A pair may be declared as a bare name or a one-element sequence, both
        of which mean the column of the same name; this is what a caller that
        has to address the data itself -- a per-tab breakdown, a log line --
        reads, rather than normalizing ``var.signals`` again at each site.
        """
        return self._norm_pairs_(self.var.signals)

    @property
    def label_pairs(self) -> tuple[tuple[str, str], ...]:
        """The label ``(slice, column)`` pairs, as :attr:`signal_pairs`; empty when there are none."""
        return self._norm_pairs_(self.var.labels or ())

    # 4. Helpers -----------------------------------------------------------

    @classmethod
    def _norm_pairs_(cls, pairs) -> tuple[tuple[str, ...], ...]:
        """Every pair normalized, a triple naming several keys expanded to one per key."""
        out = []
        for pair in pairs:
            deeper = isinstance(pair, (list, tuple)) and (
                len(pair) > 2 or (len(pair) == 2 and isinstance(pair[1], (list, tuple))))
            if not deeper:
                out.append(cls._norm_pair_(pair))
                continue
            # Lists side by side, tuples deeper -- as dataset() reads a request.
            s_name, specs = slice_spec(tuple(pair))
            for column, keys in specs:
                if keys is None:
                    out.append((s_name, column))
                    continue
                for path in (keys if isinstance(keys, list) else [keys]):
                    out.append((s_name, column, path[0] if len(path) == 1 else path))
        return tuple(out)

    @staticmethod
    def _norm_pair_(pair: Any) -> tuple[str, str]:
        if isinstance(pair, (list, tuple)):
            if len(pair) >= 2:
                return str(pair[0]), str(pair[1])
            elif len(pair) == 1:
                return str(pair[0]), str(pair[0])
        return str(pair), str(pair)

    @staticmethod
    def _as_array_(val):
        if val is None:
            return None
        if torch is not None and isinstance(val, torch.Tensor):
            return val.detach().cpu().numpy()
        arr = np.array(val)
        if hasattr(arr, 'flags') and not arr.flags.writeable:
            arr = np.copy(arr)
        return arr

    @classmethod
    def _pick_pair_(cls, row, pair, what, allow_none: bool = False):
        """The value a normalized pair -- or triple -- names in *row*."""
        return cls._pick_(row, pair[0], pair[1], what, key=pair[2] if len(pair) > 2 else None, allow_none=allow_none)

    @staticmethod
    def _pick_(row, s_name, c_name, what, key=None, allow_none: bool = False):
        """The value at ``(s_name, c_name)`` in one nested row, or a clear error.

        Exact, with no fallbacks. The previous version walked the row with
        ``next(iter(val.values()))`` when a name did not match, which meant a
        misspelled column, a renamed slice, or a row that had lost its slice
        level all silently produced *some* array -- an arbitrary one -- and
        the build wrote it as if it were the requested feature.
        """
        try:
            slice_row = row[s_name]
        except (KeyError, TypeError):
            if allow_none:
                return None
            raise KeyError(
                f"{what}: row has no slice {s_name!r}; it provides "
                f"{sorted(row) if isinstance(row, dict) else type(row).__name__}"
            ) from None
        try:
            value = slice_row[c_name]
        except (KeyError, TypeError):
            if allow_none:
                return None
            raise KeyError(
                f"{what}: slice {s_name!r} has no column {c_name!r}; it provides "
                f"{sorted(slice_row) if isinstance(slice_row, dict) else type(slice_row).__name__}"
            ) from None
        return project_column(value, key, where=f"{what}: slice {s_name!r} column {c_name!r}", allow_none=allow_none)

    def _collate_batch_(self, batch: dict, norm_pairs) -> np.ndarray:
        """Collate a ``{slice: {column: values}}`` mapping already stacked over rows.

        This is what ``data(*collator.slices(), concat=True)`` hands back, as
        opposed to the list of per-row dicts a DataLoader yields.

        One pair passes its array through untouched, so a single-signal
        collation keeps the shape the slice was written with -- which is what
        a model is then fed. Several are stacked along a new axis 1,
        mirroring the signals axis of the per-sample form.
        """
        what = f"{self.__class__.__name__}._collate_batch_"
        skip_missing = getattr(self.var, 'skip_missing', False)
        arrays = [self._as_array_(self._pick_pair_(batch, pair, what, allow_none=skip_missing)) for pair in norm_pairs]
        if skip_missing and any(a is None for a in arrays):
            if all(a is None for a in arrays):
                return None
        if len(arrays) == 1:
            return arrays[0]
        return np.stack(arrays, axis=1)

    def _collate_pairs_(self, datapoints, pairs) -> np.ndarray:
        if not pairs:
            return np.array([])

        norm_pairs = self._norm_pairs_(pairs)

        if isinstance(datapoints, dict):
            return self._collate_batch_(datapoints, norm_pairs)

        what = f"{self.__class__.__name__}._collate_pairs_"
        skip_missing = getattr(self.var, 'skip_missing', False)
        batch_items = []

        for dp in datapoints:
            dp_signals = [self._as_array_(self._pick_pair_(dp, pair, what, allow_none=skip_missing)) for pair in norm_pairs]
            if skip_missing and any(s is None for s in dp_signals):
                continue

            norm_signals = []
            for sig in dp_signals:
                if sig.ndim == 0:
                    norm_signals.append(sig.reshape(1, 1))
                elif sig.ndim == 1:
                    norm_signals.append(sig.reshape(1, -1))
                else:
                    norm_signals.append(sig)

            if len(norm_signals) == 1 and norm_signals[0].ndim >= 3:
                dp_tensor = norm_signals[0]
            elif norm_signals[0].ndim == 2 and all(x.ndim == 2 for x in norm_signals):
                try:
                    dp_tensor = np.stack(norm_signals, axis=1)
                except ValueError:
                    dp_tensor = np.concatenate(norm_signals, axis=1)
            else:
                dp_tensor = np.stack(norm_signals, axis=1)

            batch_items.append(dp_tensor)

        try:
            return np.stack(batch_items, axis=0)
        except ValueError:
            return np.concatenate(batch_items, axis=0)


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

    Inherits access to the slices of the upstream `sampletab`. Calling `dataset()`
    or `data()` with slice names present in `sampletab` seamlessly zips them in
    using the `ZipStreamingDataset` mechanism.
    """

    UPSTREAM_TABS = ('datapoint_tab',)
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
            spec={}, topics=SAME, redirect_vars={'feature_for_column': 'feature_namemap'},
            note="VAR field renamed from feature_namemap to feature_for_column"),
        Datablock.Specialization(
            spec={}, topics={'features': SLICETOPIC}, redirect_vars={'feature_for_column': 'feature_namemap'},
            note="respelled: the sentinel for DATASLICE, and feature_namemap for feature_for_column"),
    ]

    @dataclass
    class VAR(Datablock.VAR):
        datapoint_tab: Datatab
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
    #     upstream states (`datapoint_tab.declared_columns(slice)`). Under the
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
        datapoint_tab = self.var.datapoint_tab

        collator = self.var.collator
        with self.slice_writers(size_limit=self.var.shard_size_limit_bytes) as writers:
            sample_data = datapoint_tab.data(*collator.slices(), concat=True)
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
                evaluator.clear()
                del batch, result, batch_features
                gc.collect()
            del inputs
            gc.collect()
        del evaluator
        gc.collect()
        return self

    def __build_streaming__(self):
        warnings.filterwarnings("ignore", category=UserWarning, message=".*given NumPy array is not writable.*")
        gc.collect()
        evaluator = self.var.evaluator_factory.evaluator(device=self.device, log=self.log)
        datapoint_tab = self.var.datapoint_tab

        collator = self.var.collator

        dataset = datapoint_tab.dataset(*collator.slices())
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
                evaluator.clear()
                del batch_data, inputs, batch, result, batch_features
                gc.collect()
        del dataloader, dataset, evaluator
        gc.collect()
        return self

    def __len__(self) -> int:
        return len(self.var.datapoint_tab)

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
    UPSTREAM_TABS = ('datapoint_table',)
    VERSION = 1

    @dataclass
    class VAR(Datablock.VAR):
        datapoint_table: Datatable
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
        streaming: bool = False,
        dataloader_kwargs: dict | None = None,
        filter_built_tabs: bool = False,
        shared_upstream_column: 'str | None' = None,
        **kwargs,
    ):
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

    def __tab__(self, idx: int, device: str | None = None, tag=None) -> Featuretab:
        datapoint_tab = self.var.datapoint_table.tab(idx)
        spec = dict(
            datapoint_tab=datapoint_tab.quote(),
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
            tag=tag if tag is not None else datapoint_tab.tag,
        )

    def __block__(self, idx: int, **kwargs) -> Featuretab:
        return self.__tab__(idx, **kwargs)

    # 2. Declared API ------------------------------------------------------

    def validate_tab(self, i: int, **kwargs) -> bool:
        """Return whether the tab at index *i* validates."""
        signature1 = self.tab(i).var.datapoint_tab.signaturestr()
        signature2 = self.var.datapoint_table.tab(i).signaturestr()
        coherent_signatures = (signature1 == signature2)
        if not coherent_signatures:
            raise ValueError(f"tab({i}).var.datapoint_tab.signature: {signature1} " 
                             f"does not match var.datapoint_table.tab({i}): {signature2}")
        feature_dataset_len = self.tab(i).dataset().__len__()
        point_dataset_len = self.tab(i).var.datapoint_tab.dataset().__len__()
        coherent_datasets = feature_dataset_len == point_dataset_len
        if not coherent_datasets:
            raise ValueError(f"tab({i}).dataset().__len__(): {feature_dataset_len} does not match"
                             f"tab({i}).var.datapoint_table.dataset().__len__(): {point_dataset_len}"
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
        return self.var.datapoint_table.n_tabs


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

    UPSTREAM_TABS = ('featuretab',)
    VERSION = 2

    @forward_property({'bipolar': DATASLICE})
    def TOPICS(self):
        """One ``ndarray:int8`` column per encoded feature column, of the same name.

        On the class -- what a table sees of its TAB -- the slice alone.
        """
        return {'bipolar': DATASLICE(**{c: 'ndarray:int8' for c in self.feature_columns})}

    @dataclass
    class VAR(Datablock.VAR):
        featuretab: Featuretab
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
        data = self.featuretab.data(('features', list(columns)), concat=True)['features']

        from dbx.probes import normalize_features
        bipolar, counts = {}, set()
        for c in columns:
            x = np.asarray(data[c]).astype(np.float64)
            if mode is not None:
                x = np.asarray(normalize_features(x, mode))
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            if np.isnan(x).any():
                raise ValueError(f"{type(self).__name__}: column 'features.{c}' of {self.featuretab.tag!r} "
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
        return len(self.featuretab)

    # 3. Accessors ---------------------------------------------------------

    @property
    def featuretab(self) -> Featuretab:
        return self.var.featuretab

    @property
    def stats_probe(self):
        return self.var.stats_probe

    @property
    def feature_columns(self) -> list[str]:
        """The featuretab columns encoded here, in the featuretab's declared order."""
        declared = self.featuretab.declared_columns('features')
        if not declared:
            raise ValueError(
                f"{type(self).__name__}: {self.featuretab.anchorkeypath} declares no columns "
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
    UPSTREAM_TABS = ('featuretable',)
    VERSION = 2

    @dataclass
    class VAR(Datatable.VAR):
        featuretable: Featuretable = None
        stats_probe: 'FeatureStatsProbe' = None
        features: list | None = None

    # 1. Protocol and hooks ------------------------------------------------

    def __post_init__(self):
        super().__post_init__()
        if self.var.featuretable is None or self.var.stats_probe is None:
            raise ValueError(f"{type(self).__name__}: VAR.featuretable and VAR.stats_probe are both required")

    def __tab__(self, idx: int, tag=None, **kwargs) -> BipolarFeaturetab:
        featuretab = self.var.featuretable.tab(idx)
        tab_specs = (getattr(self, 'TAB_SPECIALIZATIONS', None) or getattr(self, 'TAB_SPECIALIZATION', None)
                     or getattr(self, 'BLOCK_SPECIALIZATIONS', None) or getattr(self, 'BLOCK_SPECIALIZATION', None))
        return self.TAB(
            datalake=self._datalake_,
            storage_options=self.storage_options,
            cache=getattr(self, 'cache', None),
            cache_limit=getattr(self, 'cache_limit', None),
            verbose=False,
            spec=dict(
                featuretab=dbx.quote(featuretab),
                stats_probe=dbx.quote(self.var.stats_probe),
                features=self.var.features,
            ),
            SPECIALIZATIONS=tab_specs,
            revision=self.revision,
            tag=tag if tag is not None else featuretab.tag,
        )

    def __block__(self, idx: int, **kwargs) -> BipolarFeaturetab:
        return self.__tab__(idx, **kwargs)

    # 2. Accessors ---------------------------------------------------------

    @property
    def featuretable(self) -> Featuretable:
        return self.var.featuretable

    @property
    def stats_probe(self):
        return self.var.stats_probe

    @property
    def n_tabs(self) -> int:
        return self.featuretable.n_tabs


# ═══════════════════════════════════════════════════════════════════════
#  The names these classes used to have
# ═══════════════════════════════════════════════════════════════════════

#: The names these classes used to have, as aliases -- plain assignments, so
#: the alias is the class itself. See the note at the foot of
#: :mod:`dbx.datatables`: the hash does not move, but a block of one of these
#: classes is now stored under its new name.
DatafeatureTab = FeatureTab = Featuretab
DatafeatureTable = FeatureTable = Featuretable
BipolarDatafeatureTab = BipolarFeaturetab
BipolarDatafeatureTable = BipolarFeaturetable


# ═══════════════════════════════════════════════════════════════════════
#  The module this file used to be
# ═══════════════════════════════════════════════════════════════════════

#: This file was ``dbx/datafeatures.py``. See the note at the foot of
#: :mod:`dbx.datatables` for why a module name reaches identity; it matters
#: most here, since these are the classes configured by spec and built
#: directly, so it is their own `fqcn` -- not a subclass's -- that every
#: feature artifact is stored under.
_LEGACY_MODULE = 'dbx.datafeatures'
for _obj in (
    Datacollator,
    Featuretab,
    Featuretable,
    BipolarFeaturetab,
    BipolarFeaturetable,
):
    _obj.__module__ = _LEGACY_MODULE
del _obj
