"""dbx.dataprobes — Generic feature probes and utilities for DatafeatureTables."""

from __future__ import annotations

from dataclasses import dataclass
import functools
import gc
from typing import Any

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

import dbx
from dbx.datablocks import Datablock
from dbx.datafeatures import DatafeatureTable, DatafeatureTab, Datacollator
from dbx.dataparts import (
    Logger,
    callable_executor,
    read_npz,
    read_pickle,
    read_tensor,
    write_npz,
    write_pickle,
    write_tensor,
)

NORMALIZATION_MODES = {None, "l2", "corner-l1", "corner-l2", "corner-linfty"}


def _norm_pair(pair: Any, name: str) -> tuple[str, str]:
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


class DatafeatureAffineLogisticProber:
    """Standalone logistic regression evaluator for data features.

    Provides static `evaluate_features` and `evaluate_features2`
    helpers that train/test a `LogisticRegression`
    and return `classification_report` strings.
    """

    def __init__(self, log: Logger | None = None):
        self.log = log or Logger()

    @staticmethod
    def ndarray(X: Any) -> np.ndarray:
        """Coerce X to a numpy ndarray."""
        if isinstance(X, list):
            return np.array(X)
        elif torch is not None and isinstance(X, torch.Tensor):
            return X.numpy()
        elif isinstance(X, np.ndarray):
            return X
        else:
            return np.array(X)

    @staticmethod
    def evaluate_features(Xy: tuple[Any, Any], *, training_fraction: float = 0.8,
                          fit_intercept: bool = True) -> str:
        """Train/test a LogisticRegression and return the classification report.

        *training_fraction* is the share fitted on; the remainder is scored.
        """
        features, labels = Xy
        features = DatafeatureAffineLogisticProber.ndarray(features)
        labels = DatafeatureAffineLogisticProber.ndarray(labels)
        N = len(labels)
        ntrain = int(N * training_fraction)
        perm = np.random.permutation(N)
        X_train, y_train = features[perm[:ntrain]], labels[perm[:ntrain]]
        X_test, y_test = features[perm[ntrain:]], labels[perm[ntrain:]]

        clf = LogisticRegression(fit_intercept=fit_intercept)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return classification_report(y_test, y_pred)

    @staticmethod
    def evaluate_features2(
        Xy1: tuple[Any, Any],
        Xy2: tuple[Any, Any],
        *,
        training_fraction: float = 0.8,
        fit_intercept: bool = True,
        tags: tuple[str, str] = ("(1)", "(2)"),
        log: Logger | None = None,
    ) -> tuple[str, str]:
        """Evaluate two feature sets side-by-side."""
        import datetime
        log = log or Logger()
        label1, label2 = tags
        log.verbose(f"EVALUATING features: {label1}: started at {datetime.datetime.now()}")
        report1 = DatafeatureAffineLogisticProber.evaluate_features(
            Xy1, training_fraction=training_fraction, fit_intercept=fit_intercept
        )
        log.verbose(f"EVALUATING features: {label1}: finished at {datetime.datetime.now()}")
        log.verbose(f"EVALUATING features: {label2}: started at {datetime.datetime.now()}")
        report2 = DatafeatureAffineLogisticProber.evaluate_features(
            Xy2, training_fraction=training_fraction, fit_intercept=fit_intercept
        )
        log.verbose(f"EVALUATING features: {label2}: finished at {datetime.datetime.now()}")

        rstr = f"---------- {label1} ------------\n{report1}\n---------- {label2} ------------\n{report2}"
        log.verbose(rstr)
        return report1, report2


def _pair_key(pair: tuple[str, str]) -> str:
    """The stable name a ``(slice, column)`` pair is stored under.

    The pair, not the bare column name: two slices may carry a column of the
    same name, and keying by the column alone would silently drop one of them
    -- the same collision `dataset()` keys its rows to avoid.
    """
    return f"{pair[0]}.{pair[1]}"


def _pair_array(collator: Datacollator, data: dict, pair: tuple[str, str]) -> np.ndarray:
    """One ``(slice, column)`` of a ``{slice: {column: values}}`` mapping, as an array.

    Addressed exactly, through the collator's own lookup, so a pair naming a
    column that is not there raises instead of resolving to whatever the
    mapping happened to hold first.
    """
    value = Datacollator._pick(data, pair[0], pair[1], f"dataprobes: pair {pair!r}")
    return Datacollator._as_array(value)


def signal_matrix(collator: Datacollator, data: dict, *,
                  aggregation: str | None = None,
                  normalization: str | None = None):
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
        arr = np.asarray(_pair_array(collator, data, pair), dtype=np.float32)
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
        layout.append((pair[0], pair[1], int(flat.shape[1])))

    rows = {b.shape[0] for b in blocks}
    if len(rows) != 1:
        raise ValueError(
            f"signal_matrix: signal pairs disagree on sample count: "
            f"{ {p: b.shape[0] for p, b in zip(collator.signal_pairs, blocks)} }"
        )

    X = blocks[0] if len(blocks) == 1 else np.concatenate(blocks, axis=1)
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


def label_vector(collator: Datacollator, data: dict) -> np.ndarray:
    """The single label pair as a 1-D array of per-sample labels."""
    pairs = collator.label_pairs
    if len(pairs) != 1:
        raise ValueError(
            f"label_vector: a classifier fits one label column, but the "
            f"collator declares {len(pairs)}: {pairs!r}"
        )
    y = np.asarray(_pair_array(collator, data, pairs[0]))
    y = y.reshape(len(y), -1)
    if y.shape[1] != 1:
        raise ValueError(
            f"label_vector: label pair {pairs[0]!r} has width {y.shape[1]}, "
            f"but a classifier fits a single label per sample"
        )
    return y.ravel()


class TabAffineLogisticCallable:
    """Worker callable that loads one tab's concatenated signal matrix and labels."""

    def __init__(self, probe: Any, tab_idx: int | None):
        self.probe = probe
        self.tab_idx = tab_idx

    def __call__(self):
        table = self.probe.var.feature_table
        collator = self.probe.var.collator

        block = table if self.tab_idx is None else table.tab(self.tab_idx)
        data = block.data(*collator.slices(), concat=True)
        del block

        X, layout = signal_matrix(
            collator, data,
            aggregation=self.probe.var.aggregation,
            normalization=self.probe.var.normalization,
        )
        y = label_vector(collator, data)
        del data

        if len(X) != len(y):
            raise ValueError(
                f"{type(self).__name__}: tab {self.tab_idx} has {len(X)} feature "
                f"rows and {len(y)} labels"
            )
        gc.collect()
        return {'signals': X, 'labels': y, 'layout': layout}


class DatafeatureAffineLogisticProbe(Datablock):
    """Fit a logistic classifier on the concatenation of every signal column.

    All signal pairs are treated as one vector per sample: each is flattened
    and they are laid end to end, in declaration order, so the fitted
    ``coef_`` has one coefficient per (pair, position).  That layout is stored
    as the ``columns`` topic, which is what makes a coefficient attributable
    to a feature after the fact.

    Persists the fitted classifier's ``coef_``, ``intercept_`` and
    ``classes_`` so the separating hyperplane can be inspected after building.
    """

    VERSION = 2

    TOPICS = {
        'labels': 'labels.npz',
        'features': 'features.npy',
        'columns': 'columns.pkl',
        'evaluation_report': 'evaluation_report.pkl',
        'coef': 'coef.npy',
        'intercept': 'intercept.npy',
        'classes': 'classes.npz',
    }

    @dataclass
    class VAR(Datablock.VAR):
        feature_table: DatafeatureTable | DatafeatureTab
        collator: Datacollator
        fit_intercept: bool = True
        training_fraction: float = 0.8
        # Not 'mean': aggregation collapses the axes between sample and
        # feature, which only exist for token-shaped features.  Defaulting it
        # on averaged away whatever the caller actually asked to probe.
        aggregation: str | None = None
        normalization: str | None = None  # None, 'l2', 'corner-l1', 'corner-l2', 'corner-linfty'

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
        assert self.var.normalization in NORMALIZATION_MODES, f"Unknown normalization mode: {self.var.normalization!r}"
        self.parallelization = getattr(self, 'parallelization', None) or 'inline'
        self.n_workers = getattr(self, 'n_workers', 1)
        self.work_stealing = getattr(self, 'work_stealing', getattr(self, 'works_stealing', False))
        self._prober = DatafeatureAffineLogisticProber(log=self.log)

    def _tab_results(self, tag: str, make_callable):
        table = self.var.feature_table
        n_tabs = getattr(table, 'n_tabs', 0) or 0
        executor_kwargs = dict(n_workers=self.n_workers,
                               tag=f"{tag} [{self.__class__.__name__}, n_workers={self.n_workers}]")
        if getattr(self, 'work_stealing', False):
            executor_kwargs['work_stealing'] = self.work_stealing
        executor = callable_executor(self.parallelization, **executor_kwargs)
        indices = list(range(n_tabs)) if n_tabs > 0 else [None]
        return executor.exec_callables([make_callable(i) for i in indices])

    def __build__(self):
        self.log.verbose(f"DatafeatureAffineLogisticProbe.__build__: BEGIN {self.anchorkeypath}")

        results = self._tab_results(
            "COMPUTING LOGISTIC DATA",
            lambda i: TabAffineLogisticCallable(self, i),
        )

        # Every tab must lay its columns out identically, or the rows being
        # stacked here do not describe the same feature at the same position.
        layouts = {tuple(res['layout']) for res in results}
        if len(layouts) != 1:
            raise ValueError(
                f"{self.__class__.__name__}: tabs disagree on the signal column "
                f"layout: {sorted(layouts)}"
            )
        layout = list(layouts.pop())

        X = np.concatenate([res['signals'] for res in results], axis=0)
        y = np.concatenate([res['labels'] for res in results], axis=0)
        if len(X) != len(y):
            raise ValueError(f"len(labels) != len(features): {len(y)} != {len(X)}")

        write_npz(self.path('labels', ensure_dirpath=True), labels=y)
        write_tensor(torch.from_numpy(X), self.path('features', ensure_dirpath=True))
        write_pickle(layout, self.path('columns', ensure_dirpath=True))

        N = len(y)
        ntrain = int(N * self.var.training_fraction)
        perm = np.random.permutation(N)
        X_train, y_train = X[perm[:ntrain]], y[perm[:ntrain]]
        X_test, y_test = X[perm[ntrain:]], y[perm[ntrain:]]

        self.log.verbose(
            f"FITTING LogisticRegression "
            f"(fit_intercept={self.var.fit_intercept}, "
            f"signals={self.var.collator.signal_pairs!r}, "
            f"labels={self.var.collator.label_pairs!r}, "
            f"layout={layout!r}, "
            f"aggregation={self.var.aggregation!r}, "
            f"normalization={self.var.normalization!r})"
        )
        clf = LogisticRegression(fit_intercept=self.var.fit_intercept)
        clf.fit(X_train, y_train)
        report = classification_report(y_test, clf.predict(X_test))
        self.log.verbose(f"Classification report:\n{report}")

        write_pickle(report, self.path('evaluation_report', ensure_dirpath=True))
        write_tensor(torch.from_numpy(clf.coef_), self.path('coef', ensure_dirpath=True))
        write_tensor(torch.from_numpy(clf.intercept_), self.path('intercept', ensure_dirpath=True))
        write_npz(self.path('classes', ensure_dirpath=True), classes=clf.classes_)

        self.log.verbose(f"DatafeatureAffineLogisticProbe.__build__: END {self.anchorkeypath}")
        return self

    def __read__(self, *topicpath):
        if len(topicpath) == 1 and isinstance(topicpath[0], (tuple, list)):
            topicpath = tuple(topicpath[0])
        topic = str(topicpath[0])

        if topic == 'labels':
            return read_npz(self.path('labels'), 'labels')['labels']
        if topic == 'features':
            return read_tensor(self.path('features'))
        if topic == 'columns':
            return read_pickle(self.path('columns'))
        if topic == 'evaluation_report':
            return read_pickle(self.path('evaluation_report'))
        if topic in ('coef', 'intercept'):
            return read_tensor(self.path(topic))
        if topic == 'classes':
            return read_npz(self.path('classes'), 'classes')['classes']
        raise ValueError(f"Unknown topic: {topic!r}")

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


#: Statistics computed per feature column.  Each becomes a topic holding one
#: array per column, and each has a per-tab counterpart under ``tab_<name>``.
COLUMN_STATS = ('mean', 'std', 'median', 'min', 'max', 'norm')


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
            arr = np.asarray(_pair_array(collator, data, pair))
            if not np.issubdtype(arr.dtype, np.number):
                raise TypeError(
                    f"{type(self).__name__}: column {_pair_key(pair)!r} has dtype "
                    f"{arr.dtype}, which has no mean or median. Drop it from the "
                    f"collator, or describe a numeric encoding of it instead."
                )
            arr = arr.astype(np.float64)
            if pair in signals and normalization is not None:
                arr = np.asarray(normalize_features(arr, normalization))
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            counts.add(len(arr))
            columns[_pair_key(pair)] = arr
        del data

        if len(counts) > 1:
            raise ValueError(
                f"{type(self).__name__}: columns disagree on sample count: {sorted(counts)}"
            )
        gc.collect()
        return {'columns': columns, 'n_rows': counts.pop() if counts else 0}


class DatafeatureStatsProbe(Datablock):
    """Per-column statistics for a DatafeatureTable or DatafeatureTab.

    One topic per statistic -- ``mean``, ``std``, ``median``, ``min``,
    ``max``, ``norm`` -- with the feature columns underneath it, keyed
    ``"<slice>.<column>"``.  That is the shape `dataset()` and `data()` give
    the data itself, so a statistic is addressed the way the thing it
    describes is::

        probe.read('mean')                    -> {'features.final': array(8,), ...}
        probe.read('mean', 'features.final')  -> array(8,)

    Every signal and label pair the collator declares is described, not only
    the first.  Keying by pair rather than by bare column name is what makes
    room for all of them: two slices may carry a column of the same name, and
    the old flat naming could hold only one.

    ``tab_<statistic>`` is the same statistic per tab, stacked over tabs, so a
    column's spread across tabs sits beside its spread over the whole table.
    Whole-table statistics are computed over the concatenated rows rather than
    averaged from the per-tab ones -- a mean of tab means is the table mean
    only when the tabs are equal-sized, and a median or a min never is.

    TOPICS is built per instance from the collator's pairs, as the columns are
    a property of what this probe was asked to describe rather than of the
    class.
    """

    VERSION = 2

    TOPICS = {'count': 'count.npz'}

    @dataclass
    class VAR(Datablock.VAR):
        feature_table: DatafeatureTable | DatafeatureTab
        collator: Datacollator
        normalization: str | None = None  # None, 'l2', 'corner-l1', 'corner-l2', 'corner-linfty'

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
        # The leaf is just a filename: path() renders the topic path itself as
        # directories, so the statistic and the column already name the folder.
        self.TOPICS = {
            'count': 'count.npz',
            **{name: {key: 'stat.npz' for key in self.column_keys}
               for name in COLUMN_STATS},
            **{f'tab_{name}': {key: 'stat.npz' for key in self.column_keys}
               for name in COLUMN_STATS},
        }

    @property
    def column_keys(self) -> list[str]:
        """The columns this probe describes, in the collator's declared order."""
        collator = self.var.collator
        return [_pair_key(p) for p in collator.signal_pairs + collator.label_pairs]

    def __build__(self):
        self.log.verbose(f"DatafeatureStatsProbe.__build__: BEGIN {self.anchorkeypath}")
        table = self.var.feature_table

        n_tabs = getattr(table, 'n_tabs', 0) or 0
        executor_kwargs = dict(n_workers=self.n_workers,
                               tag=f"COMPUTING TAB STATS [{self.__class__.__name__}, n_workers={self.n_workers}]")
        if getattr(self, 'work_stealing', False):
            executor_kwargs['work_stealing'] = self.work_stealing
        executor = callable_executor(self.parallelization, **executor_kwargs)
        indices = list(range(n_tabs)) if n_tabs > 0 else [None]
        results = executor.exec_callables([TabColumnStatsCallable(self, i) for i in indices])

        for key in self.column_keys:
            per_tab = [column_stats(res['columns'][key]) for res in results]
            whole = column_stats(np.concatenate([res['columns'][key] for res in results], axis=0))
            for name in COLUMN_STATS:
                write_npz(self.path(name, key, ensure_dirpath=True), stat=whole[name])
                write_npz(self.path(f'tab_{name}', key, ensure_dirpath=True),
                          stat=np.stack([tab[name] for tab in per_tab]))

        write_npz(self.path('count', ensure_dirpath=True),
                  count=np.array(sum(res['n_rows'] for res in results)))

        self.log.verbose(f"DatafeatureStatsProbe.__build__: END {self.anchorkeypath}")
        return self

    def __read__(self, *topicpath):
        if len(topicpath) == 1 and isinstance(topicpath[0], (tuple, list)):
            topicpath = tuple(topicpath[0])
        topic = str(topicpath[0])

        if topic == 'count':
            return int(read_npz(self.path('count'), 'count')['count'])
        if topic not in self.TOPICS:
            raise ValueError(
                f"Unknown topic: {topic!r}; expected one of {sorted(self.TOPICS)}"
            )
        if len(topicpath) == 1:
            return {key: self.read(topic, key) for key in self.column_keys}

        key = str(topicpath[1])
        if key not in self.column_keys:
            raise KeyError(
                f"{self.__class__.__name__}.read({topic!r}, {key!r}): no such column; "
                f"this probe describes {self.column_keys}"
            )
        return read_npz(self.path(topic, key), 'stat')['stat']

    @functools.cached_property
    def columns(self) -> list[str]:
        return self.column_keys

    @functools.cached_property
    def count(self) -> int:
        return self.read('count')
