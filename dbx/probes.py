"""dbx.probes — Generic feature probes and utilities for feature tables."""

from __future__ import annotations

from dataclasses import dataclass
import functools
import gc
import zlib
from typing import Any

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

import dbx
from dbx.datablocks import DATADICT, DATAFILE, Datablock, InvalidBlocksError, forward_property
from dbx.datatables import Datacollator
from dbx.featuretables import Featuretable, Featuretab
from dbx.datatables import DatatablePart
from dbx.dataparts import (
    callable_executor,
    read_npz,
    read_pickle,
    read_tensor,
    write_npz,
    write_pickle,
    write_tensor,
)

NORMALIZATION_MODES = {None, "l2", "corner-l1", "corner-l2", "corner-linfty"}


def _norm_pair_(pair: Any, name: str) -> tuple[str, str]:
    if isinstance(pair, (list, tuple)) and len(pair) == 2:
        return str(pair[0]), str(pair[1])
    raise ValueError(f"{name} must be specified as a (slice, column) pair of strings, e.g. ('features', 'final'), got {pair!r}")


def normalize_features(features: Any, mode: str | None) -> Any:
    """Apply optional normalization to a feature tensor or array."""
    if mode is None:
        return features

    if torch is not None and isinstance(features, torch.Tensor):
        if mode == "l2":
            return torch.nn.functional.normalize(features.float(), p=2, dim=-1)
        elif mode in ("corner-l1", "corner-l2"):
            return torch.sign(features)
        elif mode == "corner-linfty":
            abs_f = features.abs()
            idx = abs_f.argmax(dim=-1, keepdim=True)
            result = torch.zeros_like(features)
            result.scatter_(-1, idx, features.gather(-1, idx).sign())
            return result
        else:
            raise ValueError(f"Unknown normalization mode {mode!r}")
    else:
        arr = np.asarray(features)
        if mode == "l2":
            norms = np.linalg.norm(arr, ord=2, axis=-1, keepdims=True)
            norms[norms == 0] = 1.0
            return arr / norms
        elif mode in ("corner-l1", "corner-l2"):
            return np.sign(arr)
        elif mode == "corner-linfty":
            abs_f = np.abs(arr)
            idx = np.argmax(abs_f, axis=-1)
            result = np.zeros_like(arr)
            if arr.ndim == 1:
                result[idx] = np.sign(arr[idx])
            else:
                rows = np.arange(len(arr))
                result[rows, idx] = np.sign(arr[rows, idx])
            return result
        else:
            raise ValueError(f"Unknown normalization mode {mode!r}")


def _pair_key_(pair: tuple[str, str]) -> str:
    """The stable name a ``(slice, column)`` pair is stored under.

    The pair, not the bare column name: two slices may carry a column of the
    same name, and keying by the column alone would silently drop one of them
    -- the same collision `dataset()` keys its rows to avoid.
    """
    return '.'.join(_flat_parts_(pair))


def _flat_parts_(parts) -> list:
    """A pair or triple as its names in order, a key path spelled out: ``('s', 'c', ('a', 'b'))`` -> s, c, a, b."""
    out = []
    for p in parts:
        out.extend(_flat_parts_(p) if isinstance(p, tuple) else [str(p)])
    return out


def _pair_array_(collator: Datacollator, data: dict, pair: tuple[str, str], *, allow_none: bool = False) -> np.ndarray | None:
    """One ``(slice, column)`` of a ``{slice: {column: values}}`` mapping, as an array.

    Addressed exactly, through the collator's own lookup, so a pair naming a
    column that is not there raises instead of resolving to whatever the
    mapping happened to hold first.
    """
    value = Datacollator._pick_pair_(data, pair, f"probes: pair {pair!r}", allow_none=allow_none)
    return Datacollator._as_array_(value)


def signal_matrix(collator: Datacollator, data: dict, *,
                  aggregation: str | None = None,
                  normalization: str | None = None,
                  allow_none: bool = False):
    """Every signal pair flattened and concatenated into one ``(N, D)`` matrix.

    This is what "treat all features as a single vector" means concretely: the
    pairs are laid end to end along the feature axis, so column ``j`` of the
    result -- and so coefficient ``j`` of a fitted classifier -- belongs to
    exactly one pair.

    Order is ``collator.signal_pairs``, which is the order they were declared.
    That matters more here than anywhere else in the codebase: the layout is
    baked into the fitted coefficients, so a signal order that varied between
    processes would produce models whose coefficients could not be compared,
    comparable-looking reports notwithstanding, and nothing would say so.
    Hence the returned *layout*, which is stored beside the model.

    Returns
    -------
    (X, layout)
        *X* is ``(N, D)`` float32.  *layout* is one
        ``(slice, column, width)`` per pair, in the same order, with the
        widths summing to ``D``.
    """
    if not collator.signal_pairs:
        raise ValueError("signal_matrix: the collator declares no signals")

    blocks, layout = [], []
    for pair in collator.signal_pairs:
        raw = _pair_array_(collator, data, pair, allow_none=allow_none)
        if raw is None:
            if allow_none:
                return None, None
            raise ValueError(f"signal_matrix: pair {pair!r} is missing")
        arr = np.asarray(raw)
        if arr.dtype == object and allow_none and all(x is None for x in arr.ravel()):
            return None, None
        if arr.ndim == 0:
            raise ValueError(
                f"signal_matrix: pair {pair!r} is a scalar, not a per-sample column"
            )
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if aggregation is not None:
            arr = aggregate_features_np(arr, aggregation)
        flat = arr.reshape(len(arr), -1)
        blocks.append(flat)
        # A triple's column is named with its key, so the layout still says
        # which entry a coefficient came from.
        layout.append((pair[0], '.'.join(_flat_parts_(pair[1:])), int(flat.shape[1])))

    rows = {b.shape[0] for b in blocks}
    if len(rows) != 1:
        raise ValueError(
            f"signal_matrix: signal pairs disagree on sample count: "
            f"{ {p: b.shape[0] for p, b in zip(collator.signal_pairs, blocks)} }"
        )

    X = blocks[0] if len(blocks) == 1 else np.concatenate(blocks, axis=1)
    if X.dtype != object:
        X = np.asarray(normalize_features(X, normalization), dtype=np.float32)
    return X, layout


def aggregate_features_np(arr: np.ndarray, aggregation: str):
    """Collapse the axes between the sample axis and the feature axis.

    For ``(N, T, D)`` token features this averages over ``T`` and leaves
    ``(N, D)``; for an already-flat ``(N, D)`` it is the identity.  The point
    of aggregating before the flatten in `signal_matrix` is that afterwards
    there is no token axis left to average over -- only one long vector, whose
    mean is a single number per sample.
    """
    if aggregation != 'mean':
        raise ValueError(f"Unknown aggregation: {aggregation!r}")
    if arr.ndim <= 2:
        return arr
    return arr.mean(axis=tuple(range(1, arr.ndim - 1)))


def label_vector(collator: Datacollator, data: dict, *, allow_none: bool = False) -> np.ndarray | None:
    """The single label pair as a 1-D array of per-sample labels."""
    pairs = collator.label_pairs
    if len(pairs) != 1:
        raise ValueError(
            f"label_vector: a classifier fits one label column, but the "
            f"collator declares {len(pairs)}: {pairs!r}"
        )
    raw = _pair_array_(collator, data, pairs[0], allow_none=allow_none)
    if raw is None:
        if allow_none:
            return None
        raise ValueError(f"label_vector: label pair {pairs[0]!r} is missing")
    y = np.asarray(raw)
    if y.dtype == object and allow_none and all(x is None for x in y.ravel()):
        return None
    y = y.reshape(len(y), -1)
    if y.shape[1] != 1:
        raise ValueError(
            f"label_vector: label pair {pairs[0]!r} has width {y.shape[1]}, "
            f"but a classifier fits a single label per sample"
        )
    return y.ravel()


def check_probe_inputs(probe, table=None) -> None:
    """Raise `InvalidBlocksError` here, in the parent, before any worker starts, when a tab a probe will read is not valid.

    A worker reading an invalid tab fails far from the cause -- a missing shard
    index, one traceback per worker -- and says nothing of why. The tables
    asked are the ones the collator's slices are read from: *table*'s own
    (the probe's ``feature_table`` by default), and the upstream ones its
    slices route to.
    """
    table = probe.var.feature_table if table is None else table
    collator = getattr(probe.var, 'collator', None)
    # A fold routes no slices of its own: it reads its table's. Its blocks --
    # the table's tabs, and pieces of them, which are not the table's -- are
    # asked of the fold itself; an upstream table's, at the fold's tab indices.
    part = table if isinstance(table, DatatablePart) else None
    if part is not None:
        table = part.datatable
    owners = {}
    if collator is not None and hasattr(table, '_route_'):
        for owner, s_name, _ in table._route_(collator.slices(table)):
            owners.setdefault(id(owner), (owner, []))[1].append(s_name)
    else:
        owners[id(table)] = (table, list(collator.slices(table)) if collator is not None else [])
    for owner, slices in owners.values():
        if not hasattr(owner, 'valid_blocks') or not (getattr(owner, 'n_tabs', 0) or 0):
            continue
        if part is not None and owner is table:
            owner = part
            invalid = [i for i in range(part.n_tabs) if not part.valid_block(i)]
        elif part is not None:
            # A partition splits no tab of a table reading upstream, so these are whole tabs.
            invalid = [i for i in part.tab_indices if not owner.valid_block(i)]
        else:
            invalid = list(owner.valid_blocks(parallelization=probe.parallelization, n_workers=probe.n_workers,
                                              false_only=True).index)
        if invalid:
            pieces = owner is part and any(isinstance(part.tab_indices[i], dict) for i in invalid)
            raise InvalidBlocksError(owner, invalid,
                                     reader=f"{probe.anchorkeypath}: reading {slices} from its tabs"
                                            + ("; a part's pieces are built by the part: part.build()" if pieces else ""))


class TabAffineLogisticCallable:
    """Worker callable that loads one tab's concatenated signal matrix and labels."""

    def __init__(self, probe: Any, table: Any, tab_idx: int | None):
        self.probe = probe
        self.table = table
        self.tab_idx = tab_idx

    def __call__(self):
        var = self.probe.var
        collator = var.collator
        skip_missing = getattr(collator.var, 'skip_missing', False) or getattr(var, 'skip_missing', False)

        block = self.table if self.tab_idx is None else self.table.tab(self.tab_idx)
        tag = getattr(block, 'tag', None)
        data = block.data(*collator.slices(), concat=True)
        del block

        X, layout = signal_matrix(
            collator, data,
            aggregation=var.aggregation,
            normalization=var.normalization,
            allow_none=skip_missing,
        )
        y = label_vector(collator, data, allow_none=skip_missing)
        del data

        if X is None or y is None:
            if skip_missing:
                gc.collect()
                return None
            raise ValueError(
                f"{type(self).__name__}: tab {self.tab_idx} missing signals or labels "
                f"(X is None: {X is None}, y is None: {y is None})"
            )

        if len(X) != len(y):
            raise ValueError(
                f"{type(self).__name__}: tab {self.tab_idx} has {len(X)} feature "
                f"rows and {len(y)} labels"
            )

        if skip_missing:
            valid_y = np.array([v is not None and not (isinstance(v, (float, np.floating)) and np.isnan(v)) for v in y], dtype=bool)
            if X.dtype == object:
                valid_x = np.array([all(v is not None and not (isinstance(v, (float, np.floating)) and np.isnan(v)) for v in row) for row in X], dtype=bool)
            else:
                valid_x = ~np.isnan(X).any(axis=1) if np.issubdtype(X.dtype, np.floating) else np.ones(len(X), dtype=bool)
            valid = valid_x & valid_y
            n_valid = int(valid.sum())
            if n_valid == 0:
                gc.collect()
                return None
            if n_valid < len(y):
                X = X[valid]
                y = y[valid]

        if X.dtype == object:
            X = np.asarray(X, dtype=np.float32)

        # A sample of the tab's rows, seeded by the tab -- its tag, not its
        # index in this table -- so a tab gives the same rows wherever it is read.
        limit = getattr(var, 'max_rows_per_tab', None)
        if limit is not None and len(y) > limit:
            rng = np.random.default_rng([var.seed, zlib.crc32(str(tag if tag is not None else self.tab_idx).encode())])
            keep = np.sort(rng.choice(len(y), size=limit, replace=False))
            X, y = X[keep], y[keep]

        if getattr(var, 'tab_aggregation', None) == 'mean':
            labels = np.unique(y)
            if len(labels) != 1:
                raise ValueError(
                    f"{type(self).__name__}: tab {self.tab_idx} holds rows labelled "
                    f"{labels[:5].tolist()}{'...' if len(labels) > 5 else ''}; tab_aggregation='mean' "
                    f"makes the tab one sample, which can carry only one label."
                )
            X, y = X.mean(axis=0, keepdims=True), y[:1]

        gc.collect()
        return {'signals': X, 'labels': y, 'layout': layout, 'tag': tag}


class FeatureAffineLogisticProbe(Datablock):
    """Fit a logistic classifier on one table's tabs and score it on another's.

    *fit_table* and *eval_table* are what it is fitted on and scored on --
    typically two folds of a `DatatablePartition` of a feature table, dealt by
    patient and stratified by label, so neither the patients nor the slides of
    one are in the other. They must share no tab, and a probe given two that do
    refuses to build, naming them.

    All signal pairs are treated as one vector per sample: each is flattened
    and they are laid end to end, in declaration order, so the fitted
    ``coef_`` has one coefficient per (pair, position).  That layout is stored
    as the ``columns`` topic, which is what makes a coefficient attributable
    to a feature after the fact.

    A sample is a row, or with *tab_aggregation* ``'mean'`` a tab: the mean of
    its rows, labelled by the one label they share. A fold's tab may be a
    piece of a tab (`DatatabPiece`), which is a sample of its own: a partition
    stratified by the label column splits a tab whose rows carry several labels
    into pieces carrying one each, which is what makes such a tab one sample
    per label rather than a tab refused. *max_rows_per_tab* reads at
    most that many of a tab's rows, drawn with *seed* and the tab's tag -- a
    size for a fit, which moves no tab across the two tables.

    Persists the fitted classifier's ``coef_``, ``intercept_`` and
    ``classes_`` so the separating hyperplane can be inspected after building,
    and each side's features and labels under ``fit`` and ``eval``.
    """

    VERSION = 3

    TOPICS = {
        'fit': {'labels': DATADICT('labels.npz', labels='ndarray'), 'features': DATAFILE('features.npy')},
        'eval': {'labels': DATADICT('labels.npz', labels='ndarray'), 'features': DATAFILE('features.npy')},
        'columns': DATAFILE('columns.pkl', 'the feature layout: which column each feature came from'),
        'evaluation_report': DATAFILE('evaluation_report.pkl', "the classification report on eval_table"),
        'coef': DATAFILE('coef.npy'),
        'intercept': DATAFILE('intercept.npy'),
        'classes': DATADICT('classes.npz', classes='ndarray'),
    }

    @dataclass
    class VAR(Datablock.VAR):
        fit_table: Featuretable | Featuretab
        eval_table: Featuretable | Featuretab
        collator: Datacollator
        fit_intercept: bool = True
        # Not 'mean': aggregation collapses the axes between sample and
        # feature, which only exist for token-shaped features.  Defaulting it
        # on averaged away whatever the caller actually asked to probe.
        aggregation: str | None = None
        normalization: str | None = None  # None, 'l2', 'corner-l1', 'corner-l2', 'corner-linfty'
        # 'mean' makes each tab one sample: the mean of its rows, labelled by
        # the one label its rows share. A tab whose rows disagree is refused.
        tab_aggregation: str | None = None
        max_rows_per_tab: int | None = None
        seed: int = 0

    # 1. Protocol and hooks ------------------------------------------------

    def __init__(
        self,
        *args,
        parallelization: str | None = None,
        n_workers: int = 1,
        work_stealing: bool = False,
        works_stealing: bool = False,
        **kwargs,
    ):
        ws = work_stealing or works_stealing
        super().__init__(
            *args,
            parallelization=parallelization,
            n_workers=n_workers,
            work_stealing=ws,
            **kwargs,
        )

    def __post_init__(self):
        super().__post_init__()
        assert self.var.aggregation in (None, "mean"), f"Unknown aggregation: {self.var.aggregation}"
        if self.var.tab_aggregation not in (None, 'mean'):
            raise ValueError(f"{type(self).__name__}: unknown tab_aggregation {self.var.tab_aggregation!r}; "
                             f"expected None or 'mean'")
        if self.var.max_rows_per_tab is not None and int(self.var.max_rows_per_tab) < 1:
            raise ValueError(f"{type(self).__name__}: max_rows_per_tab must be positive, got {self.var.max_rows_per_tab!r}")
        assert self.var.normalization in NORMALIZATION_MODES, f"Unknown normalization mode: {self.var.normalization!r}"
        self.parallelization = getattr(self, 'parallelization', None) or 'inline'
        self.n_workers = getattr(self, 'n_workers', 1)
        self.work_stealing = getattr(self, 'work_stealing', getattr(self, 'works_stealing', False))

    def __build__(self):
        self.log.verbose(f"FeatureAffineLogisticProbe.__build__: BEGIN {self.anchorkeypath}")
        self._check_disjoint_()

        sides, layouts = {}, set()
        for side in ('fit', 'eval'):
            table = getattr(self.var, f'{side}_table')
            results = self._tab_results_(f"COMPUTING LOGISTIC INPUT DATA ({side})", table,
                                         lambda i, t=table: TabAffineLogisticCallable(self, t, i))
            results = self._kept_(side, table, results)
            layouts |= {tuple(res['layout']) for res in results}
            X = np.concatenate([res['signals'] for res in results], axis=0)
            y = np.concatenate([res['labels'] for res in results], axis=0)
            write_npz(self.path(side, 'labels', ensure_dirpath=True), labels=y)
            write_tensor(torch.from_numpy(X), self.path(side, 'features', ensure_dirpath=True))
            sides[side] = (X, y)

        # Every tab -- of both tables -- must lay its columns out identically,
        # or the rows stacked here do not describe the same feature at the same
        # position, and the eval rows are not what the classifier was fitted on.
        if len(layouts) != 1:
            raise ValueError(f"{self.__class__.__name__}: tabs disagree on the signal column layout: {sorted(layouts)}")
        layout = list(layouts.pop())
        write_pickle(layout, self.path('columns', ensure_dirpath=True))

        (X_fit, y_fit), (X_eval, y_eval) = sides['fit'], sides['eval']
        self.log.verbose(
            f"FITTING LogisticRegression "
            f"(n_fit={len(y_fit)}, n_eval={len(y_eval)}, "
            f"tab_aggregation={self.var.tab_aggregation!r}, max_rows_per_tab={self.var.max_rows_per_tab!r}, "
            f"fit_intercept={self.var.fit_intercept}, "
            f"signals={self.var.collator.signal_pairs!r}, "
            f"labels={self.var.collator.label_pairs!r}, "
            f"layout={layout!r}, "
            f"aggregation={self.var.aggregation!r}, "
            f"normalization={self.var.normalization!r})"
        )
        clf = LogisticRegression(fit_intercept=self.var.fit_intercept)
        clf.fit(X_fit, y_fit)
        report = classification_report(y_eval, clf.predict(X_eval))
        self.log.verbose(f"Classification report:\n{report}")

        write_pickle(report, self.path('evaluation_report', ensure_dirpath=True))
        write_tensor(torch.from_numpy(clf.coef_), self.path('coef', ensure_dirpath=True))
        write_tensor(torch.from_numpy(clf.intercept_), self.path('intercept', ensure_dirpath=True))
        write_npz(self.path('classes', ensure_dirpath=True), classes=clf.classes_)

        self.log.verbose(f"FeatureAffineLogisticProbe.__build__: END {self.anchorkeypath}")
        return self

    def __read__(self, *topicpath):
        if len(topicpath) == 1 and isinstance(topicpath[0], (tuple, list)):
            topicpath = tuple(topicpath[0])
        topicpath = tuple(str(t) for t in topicpath)
        topic = topicpath[0]

        if topic in ('fit', 'eval'):
            if len(topicpath) == 1:
                return {name: self.read(topic, name) for name in ('labels', 'features')}
            if topicpath[1] == 'labels':
                return read_npz(self.path(topic, 'labels'), 'labels')['labels']
            if topicpath[1] == 'features':
                return read_tensor(self.path(topic, 'features'))
        if topic == 'columns':
            return read_pickle(self.path('columns'))
        if topic == 'evaluation_report':
            return read_pickle(self.path('evaluation_report'))
        if topic in ('coef', 'intercept'):
            return read_tensor(self.path(topic))
        if topic == 'classes':
            return read_npz(self.path('classes'), 'classes')['classes']
        raise ValueError(f"Unknown topic: {'/'.join(topicpath)!r}")

    # 2. Declared API ------------------------------------------------------

    def feature_columns(self) -> list[tuple[str, str, int]]:
        """One ``(slice, column, offset)`` per column of ``coef_``.

        Expands the stored layout, so a coefficient index can be named:
        ``feature_columns()[j]`` says which pair column ``j`` came from and
        which position within it.
        """
        out = []
        for s_name, c_name, width in self.read('columns'):
            out.extend((s_name, c_name, i) for i in range(width))
        return out

    def asphericity(self) -> dict[str, float]:
        """Per-class ratio ``||intercept|| / ||coef||``."""
        coef = self.read('coef')
        intercept = self.read('intercept')
        classes = self.read('classes')
        ratios = intercept.abs() / coef.norm(dim=1)
        return {str(c): float(r) for c, r in zip(classes, ratios)}

    # 4. Helpers -----------------------------------------------------------

    @staticmethod
    def _tags_(table) -> list:
        n = getattr(table, 'n_tabs', 0) or 0
        return [table.tab(i).tag for i in range(n)] if n else [getattr(table, 'tag', None)]

    def _check_disjoint_(self) -> None:
        """Refuse a fit and an eval table that share a tab: its rows would be scored by a classifier fitted on them."""
        shared = sorted(set(self._tags_(self.var.fit_table)) & set(self._tags_(self.var.eval_table)) - {None})
        if shared:
            raise ValueError(
                f"{type(self).__name__}: fit_table and eval_table share {len(shared)} tab(s), e.g. {shared[:5]} -- "
                f"their rows would be scored by a classifier fitted on them. Give it two folds of one partition.")

    def _kept_(self, side: str, table, results) -> list:
        """The tabs' results that carry data -- warning, by tag, of those skipped for missing signals or labels."""
        kept = [res for res in results if res is not None]
        if len(kept) < len(results):
            # A warning, not info: a skipped tab is a whole slide missing from
            # the fit or the evaluation, and the report says nothing of it.
            skipped = [i for i, res in enumerate(results) if res is None]
            tags = [table.tab(i).tag for i in skipped[:10]] if getattr(table, 'n_tabs', 0) else []
            self.log.warning(
                f"{self.__class__.__name__}: {side}: skipped {len(skipped)}/{len(results)} tabs with missing "
                f"signals or labels ({len(kept)} remaining); first skipped: {tags or skipped[:10]}")
        if not kept:
            raise ValueError(f"{self.__class__.__name__}: {side}: all {len(results)} tabs were skipped "
                             f"for missing signals or labels")
        return kept

    def _tab_results_(self, tag: str, table, make_callable):
        check_probe_inputs(self, table)
        n_tabs = getattr(table, 'n_tabs', 0) or 0
        executor_kwargs = dict(n_workers=self.n_workers,
                               tag=f"{tag} [{self.__class__.__name__}, n_workers={self.n_workers}]")
        if getattr(self, 'work_stealing', False):
            executor_kwargs['work_stealing'] = self.work_stealing
        executor = callable_executor(self.parallelization, **executor_kwargs)
        indices = list(range(n_tabs)) if n_tabs > 0 else [None]
        return executor.exec_callables([make_callable(i) for i in indices])


#: The reductions over the sample axis -- each column independent of the others,
#: which is what lets the whole-table ones be taken a band of columns at a time.
#: Their names are the keys of `FeatureStatsProbe.PER_FEATURE_DATADICT`.
SAMPLE_REDUCTIONS = {
    'mean': np.mean, 'std': np.std, 'median': np.median, 'min': np.min, 'max': np.max,
}

#: Every statistic `column_stats` gives: the reductions, and ``norm`` -- one L2
#: norm per row, which is not a reduction and so is stored apart from them.
COLUMN_STATS = (*SAMPLE_REDUCTIONS, 'norm')

#: Bytes of one band of a whole-table column, concatenated across tabs.
TABLE_STATS_BAND_BYTES = 256 * 2**20


def column_stats(arr: np.ndarray) -> dict[str, np.ndarray]:
    """The `COLUMN_STATS` of one column's ``(N, ...)`` stack of values.

    Reductions run over the sample axis and so keep the shape of a single
    sample; ``norm`` is the exception, being one L2 norm per sample and so
    ``(N,)``.
    """
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return {
        'mean': np.mean(arr, axis=0),
        'std': np.std(arr, axis=0),
        'median': np.median(arr, axis=0),
        'min': np.min(arr, axis=0),
        'max': np.max(arr, axis=0),
        'norm': np.linalg.norm(arr.reshape(len(arr), -1), axis=-1),
    }


def _column_path_(pair) -> tuple[str, ...]:
    """The topic path a described pair is stored under: its names, one level each.

    ``('features', 'final')`` -> ``('features', 'final')``; a triple's key goes
    one level deeper, so ``('annotations', 'annotations', 'cohort')`` is three.
    """
    return tuple(_flat_parts_(pair))


def _nest_(paths, leaf) -> dict:
    """``{path: leaf}`` for every path, as the nested dict a hierarchical TOPICS is.

    Refuses a path that is a prefix of another: one would be a file and the
    other a directory beneath it.
    """
    tree = {}
    for path in paths:
        node = tree
        for part in path[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"column path {path!r} runs through a column described itself")
        if path[-1] in node:
            raise ValueError(f"column path {path!r} is described twice, or is a prefix of another")
        node[path[-1]] = leaf
    return tree


class TabColumnStatsCallable:
    """Worker callable returning one tab's per-column values, ready to describe."""

    def __init__(self, probe: Any, tab_idx: int | None):
        self.probe = probe
        self.tab_idx = tab_idx

    def __call__(self):
        table = self.probe.var.feature_table
        collator = self.probe.var.collator
        normalization = self.probe.var.normalization
        signals = set(collator.signal_pairs)

        block = table if self.tab_idx is None else table.tab(self.tab_idx)
        data = block.data(*collator.slices(), concat=True)
        del block

        columns, counts = {}, set()
        for pair in collator.signal_pairs + collator.label_pairs:
            arr = np.asarray(_pair_array_(collator, data, pair))
            if not np.issubdtype(arr.dtype, np.number):
                raise TypeError(
                    f"{type(self).__name__}: column {_pair_key_(pair)!r} has dtype "
                    f"{arr.dtype}, which has no mean or median. Drop it from the "
                    f"collator, or describe a numeric encoding of it instead."
                )
            arr = arr.astype(np.float64)
            if pair in signals and normalization is not None:
                arr = np.asarray(normalize_features(arr, normalization))
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            counts.add(len(arr))
            columns[_column_path_(pair)] = arr
        del data

        if len(counts) > 1:
            raise ValueError(
                f"{type(self).__name__}: columns disagree on sample count: {sorted(counts)}"
            )
        # This tab's own statistics, here in the worker rather than in the
        # build's loop over every tab afterwards: they then run in parallel.
        stats = {path: column_stats(arr) for path, arr in columns.items()}
        gc.collect()
        return {'columns': columns, 'stats': stats, 'n_rows': counts.pop() if counts else 0}


class FeatureStatsProbe(Datablock):
    """Per-column statistics for a Featuretable or Featuretab.

    Every signal and label pair the collator declares is described, each under
    its own path -- the pair's names, one level each -- in three topics::

        stats/<slice>/<column>/stats.npz          PER_FEATURE_DATADICT
        tab_stats/<slice>/<column>/tab_stats.npz  PER_TAB_DATADICT
        norm/<slice>/<column>/norm.npz            NORM_DATADICT

    A statistic is addressed the way the data it describes is::

        probe.read('stats', 'features', 'final')   -> {'mean': array(8,), 'median': ..., ...}
        probe.stat('median', ('features', 'final'))                -> array(8,)
        probe.stat('median', ('features', 'final'), per_tab=True)  -> array(n_tabs, 8)

    Whole-table statistics are computed over the concatenated rows rather than
    averaged from the per-tab ones -- a mean of tab means is the table mean
    only when the tabs are equal-sized, and a median or a min never is.

    Signal columns are described after `normalize_features` with
    ``VAR.normalization``, one column at a time; labels as they are. A block
    that thresholds against these statistics (`BipolarFeaturetab`) must
    normalize its values the same way, and reads the mode from here to do so.

    TOPICS is built per instance from the collator's pairs, as the columns are
    a property of what this probe was asked to describe rather than of the
    class.
    """

    VERSION = 3

    #: What is stored for each described column, over the whole table: one
    #: array per statistic, each the shape of a single sample.
    PER_FEATURE_DATADICT = DATADICT(
        'stats.npz', mean='ndarray', std='ndarray', median='ndarray', min='ndarray', max='ndarray')
    #: The same statistics per tab, stacked in tab order: ``(n_tabs, ...)``.
    PER_TAB_DATADICT = DATADICT(
        'tab_stats.npz', mean='ndarray', std='ndarray', median='ndarray', min='ndarray', max='ndarray')
    #: One L2 norm per row, concatenated in tab order, and each tab's row
    #: count, which splits them back into tabs.
    NORM_DATADICT = DATADICT('norm.npz', norm='ndarray', tab_counts='ndarray')

    @forward_property({'count': DATADICT('count.npz', count='ndarray')})
    def TOPICS(self):
        """``count``, and the three per-column topics under every described column's path.

        On the class, ``count`` alone: the columns are the collator's to say.
        """
        paths = self.column_paths
        return {
            'count': DATADICT('count.npz', count='ndarray'),
            'stats': _nest_(paths, self.PER_FEATURE_DATADICT),
            'tab_stats': _nest_(paths, self.PER_TAB_DATADICT),
            'norm': _nest_(paths, self.NORM_DATADICT),
        }

    @dataclass
    class VAR(Datablock.VAR):
        feature_table: Featuretable | Featuretab
        collator: Datacollator
        normalization: str | None = None  # None, 'l2', 'corner-l1', 'corner-l2', 'corner-linfty'

    # 1. Protocol and hooks ------------------------------------------------

    def __init__(
        self,
        *args,
        parallelization: str | None = None,
        n_workers: int = 1,
        work_stealing: bool = False,
        works_stealing: bool = False,
        **kwargs,
    ):
        ws = work_stealing or works_stealing
        super().__init__(
            *args,
            parallelization=parallelization,
            n_workers=n_workers,
            work_stealing=ws,
            **kwargs,
        )

    def __post_init__(self):
        super().__post_init__()
        assert self.var.normalization in NORMALIZATION_MODES, f"Unknown normalization mode: {self.var.normalization!r}"
        self.parallelization = getattr(self, 'parallelization', None) or 'inline'
        self.n_workers = getattr(self, 'n_workers', 1)
        self.work_stealing = getattr(self, 'work_stealing', getattr(self, 'works_stealing', False))

    def __build__(self):
        self.log.verbose(f"FeatureStatsProbe.__build__: BEGIN {self.anchorkeypath}")
        check_probe_inputs(self)
        table = self.var.feature_table

        n_tabs = getattr(table, 'n_tabs', 0) or 0
        executor_kwargs = dict(n_workers=self.n_workers,
                               tag=f"COMPUTING TAB STATS [{self.__class__.__name__}, n_workers={self.n_workers}]")
        if getattr(self, 'work_stealing', False):
            executor_kwargs['work_stealing'] = self.work_stealing
        executor = callable_executor(self.parallelization, **executor_kwargs)
        indices = list(range(n_tabs)) if n_tabs > 0 else [None]
        self.log.info(f"{type(self).__name__}: computing per-tab stats of "
                      f"{len(self.column_paths)} column(s) over {len(indices)} tab(s)")
        results = executor.exec_callables([TabColumnStatsCallable(self, i) for i in indices])

        n_rows = sum(res['n_rows'] for res in results)
        held = sum(arr.nbytes for res in results for arr in res['columns'].values())
        self.log.info(f"{type(self).__name__}: {len(results)} tab(s), {n_rows} row(s) in hand "
                      f"({held / 2**30:.2f} GiB); computing whole-table stats")
        whole_by_path = self._table_stats_(results)

        from tqdm import tqdm
        tab_counts = np.array([res['n_rows'] for res in results])
        for path in tqdm(self.column_paths, desc=f"WRITING STATS [{type(self).__name__}]"):
            whole = whole_by_path[path]
            write_npz(self.path('stats', *path, ensure_dirpath=True),
                      **{name: whole[name] for name in self.PER_FEATURE_DATADICT.schema})
            write_npz(self.path('tab_stats', *path, ensure_dirpath=True),
                      **{name: np.stack([res['stats'][path][name] for res in results])
                         for name in self.PER_TAB_DATADICT.schema})
            write_npz(self.path('norm', *path, ensure_dirpath=True),
                      norm=whole['norm'], tab_counts=tab_counts)

        write_npz(self.path('count', ensure_dirpath=True), count=np.array(n_rows))
        self.log.info(f"{type(self).__name__}: stats written")

        self.log.verbose(f"FeatureStatsProbe.__build__: END {self.anchorkeypath}")
        return self

    def __read__(self, *topicpath):
        if len(topicpath) == 1 and isinstance(topicpath[0], (tuple, list)):
            topicpath = tuple(topicpath[0])
        topic = str(topicpath[0])
        path = tuple(str(p) for p in topicpath[1:])

        if topic == 'count':
            return int(read_npz(self.path('count'), 'count')['count'])
        marker = self._stat_marker_(topic)
        if not path:
            return {p: self.read(topic, *p) for p in self.column_paths}
        self._check_column_path_(path, f"read({topic!r}, {', '.join(map(repr, path))})")
        return read_npz(self.path(topic, *path), *marker.schema)

    # 2. Declared API ------------------------------------------------------

    def stat(self, name: str, column, *, per_tab: bool = False) -> np.ndarray | list:
        """One statistic of one described column.

        *column* is the pair as the collator declares it, ``('features',
        'final')``. Whole-table by default; ``per_tab=True`` gives the tabs'
        values stacked, ``(n_tabs, ...)``. ``'norm'`` is per row: the whole
        table's rows in tab order, or with ``per_tab=True`` one array per tab.
        Only the one array asked for is read from its file.
        """
        if not isinstance(column, (tuple, list)):
            raise TypeError(f"{type(self).__name__}.stat: column must be the declared pair, "
                            f"e.g. ('features', 'final'), got {column!r}")
        path = _column_path_(column)
        self._check_column_path_(path, f"stat({name!r}, {column!r})")
        if name == 'norm':
            if not per_tab:
                return read_npz(self.path('norm', *path), 'norm')['norm']
            d = read_npz(self.path('norm', *path), 'norm', 'tab_counts')
            return np.split(d['norm'], np.cumsum(d['tab_counts'])[:-1])
        if name not in self.PER_FEATURE_DATADICT.schema:
            raise KeyError(f"{type(self).__name__}.stat: no statistic {name!r}; "
                           f"expected one of {list(COLUMN_STATS)}")
        topic = 'tab_stats' if per_tab else 'stats'
        return read_npz(self.path(topic, *path), name)[name]

    # 3. Accessors ---------------------------------------------------------

    @property
    def column_paths(self) -> list[tuple[str, ...]]:
        """The columns this probe describes, as topic paths, in the collator's declared order."""
        collator = self.var.collator
        return [_column_path_(p) for p in collator.signal_pairs + collator.label_pairs]

    @functools.cached_property
    def columns(self) -> list[tuple[str, ...]]:
        return self.column_paths

    @functools.cached_property
    def count(self) -> int:
        return self.read('count')

    # 4. Helpers -----------------------------------------------------------

    def _stat_marker_(self, topic: str):
        markers = {'stats': self.PER_FEATURE_DATADICT, 'tab_stats': self.PER_TAB_DATADICT,
                   'norm': self.NORM_DATADICT}
        if topic not in markers:
            raise ValueError(f"Unknown topic: {topic!r}; expected one of {['count', *markers]}")
        return markers[topic]

    def _check_column_path_(self, path: tuple, where: str) -> None:
        if tuple(path) not in self.column_paths:
            raise KeyError(
                f"{type(self).__name__}.{where}: no such column; "
                f"this probe describes {self.column_paths}"
            )

    def _table_stats_(self, results) -> dict[tuple, dict[str, np.ndarray]]:
        """The whole-table `column_stats` of every column, a band of columns at a time.

        Concatenating a column across all tabs and reducing it is a full copy of
        the table's features -- and ``np.median`` takes another to partition --
        which on a large table is tens of GB, and is where the build sat, with
        no progress bar, after the last tab. Every reduction here runs over the
        sample axis, independently per column, so reducing one band of columns
        at a time gives the same values holding only that band. ``norm`` is per
        sample, so it is the tabs' own, joined.
        """
        from tqdm import tqdm
        plans = []
        for path in self.column_paths:
            parts = [res['columns'][path] for res in results]
            n_rows = sum(len(p) for p in parts)
            width = parts[0].shape[1]
            per_col = n_rows * parts[0].dtype.itemsize * int(np.prod(parts[0].shape[2:], dtype=np.int64))
            band = max(1, TABLE_STATS_BAND_BYTES // max(1, per_col))
            plans.append((path, parts, width, band))
        steps = sum(-(-width // band) for _, _, width, band in plans)

        out = {}
        with tqdm(total=steps, desc=f"COMPUTING TABLE STATS [{self.__class__.__name__}]") as bar:
            for path, parts, width, band in plans:
                pieces = {name: [] for name in SAMPLE_REDUCTIONS}
                for c0 in range(0, width, band):
                    chunk = np.concatenate([p[:, c0:c0 + band] for p in parts], axis=0)
                    for name, reduce in SAMPLE_REDUCTIONS.items():
                        pieces[name].append(reduce(chunk, axis=0))
                    del chunk
                    bar.update(1)
                out[path] = {name: np.concatenate(pieces[name], axis=0) for name in SAMPLE_REDUCTIONS}
                out[path]['norm'] = np.concatenate([res['stats'][path]['norm'] for res in results])
        return out


# The statistics are spelled out in the DATADICTs, as documentation, and
# computed by SAMPLE_REDUCTIONS: the two must name the same ones.
for _marker in (FeatureStatsProbe.PER_FEATURE_DATADICT, FeatureStatsProbe.PER_TAB_DATADICT):
    if tuple(_marker.schema) != tuple(SAMPLE_REDUCTIONS):
        raise RuntimeError(f"{_marker!r} declares {tuple(_marker.schema)}, "
                           f"but SAMPLE_REDUCTIONS computes {tuple(SAMPLE_REDUCTIONS)}")
del _marker
