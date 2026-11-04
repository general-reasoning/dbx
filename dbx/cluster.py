"""Running on a cluster: a pipeline's call, or a shell command, run somewhere else, and journaled here.

A *target* is where a run goes: a cluster -- a Kubernetes cluster with Kueue (`dbx.kube.Cluster`),
and later a Slurm partition of one. It is a value, passed where it is used -- nothing is registered:

    LAMBDA = dbx.kube.Cluster('lambda', namespace=..., queue=..., ...)
    my_pipeline('preset').runner(LAMBDA).build_tree()

A name works too, for a `dbx.pprint` one-liner, a console script or a new shell: `cluster('lambda')`
looks it up, each time, among the clusters the installed packages name as entry points of group
`dbx.clusters` -- each a Target, or a dict or list of them -- imported for the lookup and kept nowhere:

    [project.entry-points."dbx.clusters"]
    myproject = "myproject.clusters:CLUSTERS"

(The group was `dbx.targets` until 2026-11-04; it is still read, with a DeprecationWarning.)

`Pipeline.runner(target, **options)` hands out the pipeline's runs as a `Runner` of the target's
kind -- `target.RUNNER`, e.g. `dbx.kube.KubeRunner` -- whose `build_tree()` (or `ship('<method>',
...)`) runs there: `dbx.eval('<the pipeline's quote>').entry.<method>(...)`, under `dbx.pprint`,
with the entry's datalake as `DBX_DATALAKE`. It returns a `Run`: `status`, `wait`, `logs`,
`cancel`. (`Pipeline.submit` is the older spelling.)

The run is journaled on the pipeline, in the `PipeJournal` of the submitting datalake, each entry
with the pipeline's `entrypath` and the run's `cluster` and `run` (its id):

    submit:BEGIN    before it is launched; its note (a file) the whole record, as JSON: the kind and
                    the target itself (`target_spec`), the run's id, the command and expression, the
                    submitting session, and what the kind adds -- a Kube Job's manifest, image,
                    revision, synced workdirs
    submit:END      when `Run.wait` sees it complete
    submit:ERROR    when launching it fails, or `Run.wait` sees it fail (with its log tail)
    submit:CANCEL   `Run.cancel`

A run's Job sets `DBX_CLUSTER` and `DBX_RUN`, so every build-journal entry written in it -- each
datablock's `build:start`, `build:end` -- names the run as well. A Runner -- `Pipeline.runner`, or
`runner(what)` of a datablock, a Datablock class or anchor, or a Pipeline class -- reads them back
(`runs`, `rundash`, `cancel`), each run as its Run, its target rebuilt from the record -- to
`status`, `logs`, `wait` on or `cancel` from another process, a later session. Runs submitted
before were journaled on the pipeline's entry, in its own journal, and are read there too.

The run's own record is the target's datalake (the entry's, `DBX_DATALAKE`): the call runs as
`dbx.pprint --capture-output`, under a session chosen here (`DBX_JOURNAL_SESSION`, recorded as
`run_session`), so its exec-journal row, every build-journal entry it writes (each with its `log`)
and its captured stdout and stderr are under that session there. `Run.command()` is that row, an
`ExecJournalEntry`, read from the record's datalake or the *datalake* given -- here on a cluster that
mounts it, on the cluster's own machines otherwise, or by URL (`Cluster.url`) -- and
`Run.command().output()` prints the capture: the journal is where the output is, and the one way to it.

`dry_run`, an option of every kind: True prepares the record and launches nothing; a kind may check
more without launching (`'server'`: Kube's API server validates the Job); `'cluster'` launches a
small run on the target that only evaluates the pipeline's quote there and `verify`s the entry --
every import, the venv, the revision, the workdirs, the datalake, the same hash -- and builds nothing.
"""
import ast
import dataclasses
import datetime
import functools
import importlib
import json
import os
import re
import sys
import time
import uuid
import warnings
from importlib import metadata

import pandas as pd

from . import databits
from .databits import default_datalake
from .journals import BuildJournal, PipeJournal, execjournal


__all__ = ['Target', 'Run', 'Runner', 'Submit', 'GROUP', 'cluster', 'clusters', 'runner', 'runs', 'rundash', 'verify']


# The entry-point group naming the installed packages' clusters, and its name before 2026-11-04.
GROUP = 'dbx.clusters'
_OLD_GROUP_ = 'dbx.targets'


class Target:
    """Where a run goes. A kind subclasses it -- a dataclass, so the journal can record it -- with its
    `MODE`, its `Runner` subclass `RUNNER`, and a `name`."""
    MODE: str = None
    RUNNER: type = None
    name: str


class Run:
    """A launched run: its `status`, `wait`, `logs`, `cancel`. `id` is None when nothing was launched (dry run, or `skipped`).

    *on* is what its submission is journaled on: the pipeline that shipped it (in the `PipeJournal`),
    or the datablock a bare command was given (`dbx.kube.submit(..., block=)`); None for neither.

    A kind implements `_status_` -- 'pending', 'running', 'complete', 'failed' -- `_logs_` and `_cancel_`.
    """

    def __init__(self, target: Target, id: str | None, record: dict, *, on=None, skipped: str | None = None):
        self.target, self.id, self.record, self.on, self.skipped = target, id, record, on, skipped
        self._closed_ = False
        # The journaled `(event, datetime)` that closed it, or None; set by `Runner.runs`.
        self.closing = None
        # A function giving what to journal its close on, when *on* is None; set by `Runner.runs`.
        self._journal_on_ = None

    def __repr__(self):
        what = self.id or (f"skipped: {self.skipped}" if self.skipped else 'dry run')
        return f"{type(self).__name__}({self.target.MODE}:{self.target.name}, {what})"

    def status(self) -> str:
        """'pending', 'running', 'complete', 'failed'; 'skipped' or 'dry-run' when nothing was launched."""
        if self.id is None:
            return 'skipped' if self.skipped else 'dry-run'
        return self._status_()

    def logs(self, tail: int = -1) -> str:
        """The run's log: the last *tail* lines, all when -1."""
        return '' if self.id is None else self._logs_(tail)

    def wait(self, *, poll: float = 30, timeout: float | None = None, verbose: bool | None = None) -> bool:
        """Block until the run completes (True), or *timeout* passes (False); raise RuntimeError with its log tail if it fails.

        Journals `submit:END` or `submit:ERROR` on what it was submitted on, once. *verbose* (by default
        when stderr is a terminal) says on stderr what the run is doing, as it changes: its status with
        the elapsed time, why it is pending (`reason`, where the kind says), and its log's latest line.
        Interrupting the wait leaves the run running: `Runner.runs(open=True)[-1].wait()` takes it up again.
        """
        if self.id is None:
            return True
        verbose = sys.stderr.isatty() if verbose is None else verbose
        start, said = time.monotonic(), {}
        while timeout is None or time.monotonic() - start < timeout:
            s = self._status_()
            if verbose:
                self._say_(s, time.monotonic() - start, said)
            if s == 'complete':
                self._close_('submit:END', f"{self._where_()} complete")
                return True
            if s == 'failed':
                tail = self._logs_(40)
                self._close_('submit:ERROR', f"{self._where_()} failed\n{tail}")
                raise RuntimeError(f"{self._where_()} failed; its log ends:\n{tail}")
            time.sleep(poll)
        return False

    def _say_(self, status: str, elapsed: float, said: dict) -> None:
        """Print what changed since *said* -- the status (and why it is pending), the log's latest line -- and remember it."""
        clock = time.strftime('%H:%M:%S')
        reason = None
        if status == 'pending' and hasattr(self, 'reason'):
            try:
                reason = self.reason()
            except Exception as e:
                reason = f"(no reason: {str(e).splitlines()[0][:80]})"
        if (status, reason) != said.get('status'):
            why = f": {reason}" if reason else ''
            print(f"[run] {clock} {self._where_()}: {status} after {elapsed / 60:.1f} min{why}", file=sys.stderr,
                  flush=True)
            said['status'] = (status, reason)
        if status == 'running':
            try:
                line = next((ln for ln in reversed(self._logs_(5).splitlines()) if ln.strip()), '')
            except Exception:
                line = ''
            if line and line != said.get('line'):
                print(f"[run] {clock}   {line[:200]}", file=sys.stderr, flush=True)
                said['line'] = line

    def cancel(self) -> None:
        """Stop the run; journals `submit:CANCEL`."""
        if self.id is not None:
            self._cancel_()
            self._close_('submit:CANCEL', f"{self._where_()} cancelled")

    def command(self, *, datalake: str | None = None, storage_options: dict | None = None):
        """The run's own exec-journal row (an `ExecJournalEntry`), under its `run_session`, in the record's datalake or *datalake*.

        `.output()` on it prints what the run printed. The row is written when the run ends.
        """
        session = self.record.get('run_session')
        if not session:
            raise LookupError(f"{self!r} was not given a session of its own to run under: no row to find")
        where = dict(datalake=datalake or self.record.get('datalake'), storage_options=storage_options,
                     session=f"^{session}$")
        if len(execjournal(**where)) == 0:
            raise LookupError(f"No command of session {session} in {where['datalake']}: not over yet (its row is "
                              f"written when it ends), or that is not the datalake it ran in")
        return execjournal(iloc=0, **where)

    def _where_(self) -> str:
        return f"{self.target.MODE}:{self.target.name} {self.id}"

    def _close_(self, event: str, note: str) -> None:
        if self.on is None and self._journal_on_ is not None and not self._closed_:
            self.on = self._journal_on_()
        if self.on is not None and not self._closed_:
            _journal_(self.on, event, note=note, inline_note=True, cluster=self.target.name, run=self.id)
        self._closed_ = True

    def _status_(self) -> str:
        raise NotImplementedError

    def _logs_(self, tail: int) -> str:
        raise NotImplementedError

    def _cancel_(self) -> None:
        raise NotImplementedError


class Runner:
    """The runs of a pipeline, a datablock or a Datablock class: what launches them, and what lists, watches and cancels them.

    `Pipeline.runner(target, **options)` hands out the pipeline's, of the target's kind --
    `target.RUNNER`, e.g. `dbx.kube.KubeRunner` -- and `runner(what, target)` anyone's:

        r = my_pipeline('preset').runner(LAMBDA, gpus=8)
        r.build_tree()                          # a Run, of dbx.eval('<the quote>').entry.build_tree() there
        r.ship('try_specialize_tree', ...)      # any other method of the entry
        r.runs(), r.rundash()                   # its runs on LAMBDA, from the entry's journal
        r.cancel()                              # those still pending or running

    `ship(method, ...)` launches `dbx.eval('<the pipeline's quote>').entry.<method>(...)` under
    `dbx.pprint`, with the entry's datalake as `DBX_DATALAKE`: the quote rebuilds the same pipeline --
    the same hashes, the same operational kwargs -- so the run builds the very datablocks a call
    here would. The arguments must be literals (`ast.literal_eval(repr(x)) == x`): they travel as
    text. `build` and `build_tree` of an entry already valid here launch nothing -- a Run `skipped`
    -- unless `force`. Only a pipeline's Runner with a target launches.

    Which runs `runs`, `rundash` and `cancel` see: a pipeline's are those it shipped (in the
    `PipeJournal`, under its hash); a datablock's those whose Job wrote its build entries, and those
    of the pipelines it is the entry of; a Datablock class's or an anchor's, those of every
    datablock of it, whatever its hash; a Pipeline class's, those of every pipeline of it. With
    *adopted*, a pipeline's and a datablock's add the runs of the builds it adopted through a
    specialization. Only those on the target, when there is one.

    A kind subclasses it: `_check_` refuses options it does not take, at construction; `_prepare_`
    makes the record (the run's id among it); `_stage_` gets what the run needs in place;
    `_create_` launches it; `RUN` is its Run. `dry_run` -- an option of every kind -- prepares the
    record and launches nothing; a kind may check more for `dry_run='server'`, in `_dry_run_`.
    *what* is None for a bare command, launched with `_launch_`.
    """
    RUN: type = Run

    # 1. Protocol ---------------------------------------------------------------------------- #

    def __init__(self, what=None, target: 'str | Target | None' = None, *, mode: str | None = None,
                 force: bool = False, datalake: str | None = None, **options):
        from .pipelines import Pipeline  # pipelines imports this module
        self._what_, self._force_ = what, force
        self._target_ = _target_of_(target, mode) if target is not None else None
        self._pipeline_ = what if isinstance(what, Pipeline) else None
        self._dblk_ = what.entry if self._pipeline_ is not None else (
            what if what is not None and not isinstance(what, (str, type)) else None)
        self._datalake_ = datalake or getattr(self._dblk_, 'datalake', None)
        if self._target_ is not None:
            self._check_(options)
        self._options_ = options

    def __repr__(self):
        what = self._pipeline_.quote() if self._pipeline_ is not None else (
            self._dblk_.anchorkeypath if self._dblk_ is not None else self._anchor_())
        return f"{type(self).__name__}({what!r}, {self.target.name if self.target else None!r})"

    # 2. Declared API ------------------------------------------------------------------------ #

    def ship(self, method: str, *args, **kwargs) -> Run:
        """Launch `entry.<method>(*args, **kwargs)` on the target; its Run."""
        if self._pipeline_ is None or self._target_ is None:
            raise TypeError(f"{self!r} launches nothing: only a pipeline's Runner with a target does "
                            f"(`Pipeline.runner(target)`)")
        entry = self._pipeline_.entry
        if not callable(getattr(type(entry), method, None)):
            raise AttributeError(f"{type(entry).__name__} has no method {method!r} to ship")
        bad = [repr(a) for a in (*args, *kwargs.values()) if not _literal_(a)]
        if bad:
            raise TypeError(f"{type(self).__name__}.ship({method!r}): {', '.join(bad)} would not survive the trip "
                            f"as text; pass literals (str, numbers, None, and lists, dicts, tuples of them)")
        options = dict(self._options_)
        check = options.get('dry_run') == 'cluster'
        if method in ('build', 'build_tree') and not self._force_ and not check and entry.valid():
            return self.RUN(self._target_, None, {}, on=self._pipeline_, skipped=f"{entry.anchorkeypath} is valid here")
        call = ', '.join([repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()])
        expression = f"dbx.eval({self._pipeline_.quote()!r}).entry.{method}({call})"
        options.setdefault('gpus', getattr(self._pipeline_, 'n_gpus', None) or 0)
        options.setdefault('name', f"{type(self._pipeline_).__name__.removesuffix('Pipeline')}-{method}")
        if check:  # The call is recorded, not made: the run evaluates the quote and verifies the entry.
            options.pop('dry_run')
            options = {**options, **self._check_options_(), 'name': f"{options['name']}-check"}
            options['env'] = {'DBX_SUBMIT_CALL': expression, **options.get('env', {})}
            expression = f"dbx.cluster.verify(dbx.eval({self._pipeline_.quote()!r}).entry, {entry.hash!r})"
        options['env'] = {'DBX_DATALAKE': entry.datalake, 'DBX_SUBMIT_EXPRESSION': expression,
                          'DBX_PARENT_SESSION': _session_(), 'DBX_JOURNAL_SESSION': str(uuid.uuid4()),
                          'DBX_JOURNAL_DATALAKE': entry.datalake, **options.get('env', {})}
        return self._launch_('dbx.pprint --capture-output "$DBX_SUBMIT_EXPRESSION"', on=self._pipeline_,
                             expression=expression, **options)

    def build(self, *args, **kwargs) -> Run:
        """`ship('build', ...)`."""
        return self.ship('build', *args, **kwargs)

    def build_tree(self, *args, **kwargs) -> Run:
        """`ship('build_tree', ...)`."""
        return self.ship('build_tree', *args, **kwargs)

    def runs(self, *, open: bool | None = None, adopted: bool = False) -> list:
        """The runs in scope (see the class), oldest first, as their Runs, from the journal; only those still open, or only those closed, by *open*.

        A run whose `submit:END`, `submit:ERROR` or `submit:CANCEL` is journaled is closed: its
        `wait` journals nothing more. Its target is rebuilt from its record.
        """
        out = [run for run, _ in self._journaled_(adopted)]
        return out if open is None else [r for r in out if r._closed_ != open]

    def cancel(self, *, adopted: bool = False) -> list:
        """Cancel the runs in scope still pending or running (asking the cluster); the Runs cancelled.

        Each is journaled `submit:CANCEL`. A run whose Job is over, or gone, is left as it is.
        """
        out = []
        for run in self.runs(open=True, adopted=adopted):
            if _live_status_(run) in ('pending', 'running'):
                run.cancel()
                out.append(run)
        return out

    def rundash(self, *, adopted: bool = False, live: bool = True) -> pd.DataFrame:
        """The runs in scope as a table, newest first.

        One row per run: `run, cluster, entry, method, status, submitted, ended, elapsed, event`;
        `entry` is the key of the pipeline's entry it built.
        `status` is the journal's for a closed run (`complete`, `failed`, `cancelled`), and with
        *live* the cluster's for an open one (`pending`, `running`, `complete`, `failed`, or `gone`
        once the Job is deleted); `event` is the journaled closing event, if any. `elapsed` runs
        from `submitted` to `ended`, or to now while open.
        """
        rows = []
        now = pd.Timestamp.now()
        for run, row in self._journaled_(adopted):
            closing = run.closing
            status = _CLOSED_STATUS_.get(closing[0]) if closing else (_live_status_(run) if live else 'open')
            submitted = pd.Timestamp(run.record.get('submitted') or row['datetime'])
            ended = pd.Timestamp(closing[1]) if closing else None
            m = re.search(r"\.entry\.(\w+)\(|dbx\.cluster\.(verify)\(", run.record.get('expression') or '')
            until = ended if ended is not None else (now if status in ('pending', 'running', 'open') else None)
            rows.append(dict(run=run.id, cluster=run.target.name, entry=os.path.basename(str(run.entrypath).rstrip('/')),
                             method=(m.group(1) or m.group(2)) if m else None, status=status,
                             submitted=submitted, ended=ended,
                             elapsed=_round_(until - submitted) if until is not None else None,
                             event=closing[0] if closing else None))
        cols = ['run', 'cluster', 'entry', 'method', 'status', 'submitted', 'ended', 'elapsed', 'event']
        df = pd.DataFrame(rows, columns=cols)
        return df.sort_values('submitted', ascending=False).reset_index(drop=True)

    # 3. Accessors --------------------------------------------------------------------------- #

    @property
    def pipeline(self):
        """The pipeline whose runs these are, or None."""
        return self._pipeline_

    @property
    def entry(self):
        """The datablock whose runs these are -- the pipeline's entry -- or None for a class's or an anchor's."""
        return self._dblk_

    @property
    def target(self) -> 'Target | None':
        return self._target_

    # 4. Helpers ----------------------------------------------------------------------------- #

    def _launch_(self, command: str, *, on=None, expression: str | None = None, **options) -> Run:
        """Launch *command* with *options* (over those given at construction); journaled on *on* -- a pipeline or a datablock -- when given."""
        options = {**self._options_, **options}
        dry_run = options.pop('dry_run', False)
        dry_run = False if dry_run == 'cluster' else dry_run  # A check is a run: ship made it one.
        record = dict(executor=self._target_.MODE, target=self._target_.name, target_spec=_spec_(self._target_),
                      command=command, expression=expression,
                      run_session=(options.get('env') or {}).get('DBX_JOURNAL_SESSION'),
                      submitted=datetime.datetime.now().isoformat())
        record.update(self._prepare_(command, **options))
        if dry_run:
            self._dry_run_(record, dry_run)
            return self.RUN(self._target_, None, record)
        record.update(self._stage_(record, **options))
        where = dict(cluster=self._target_.name, run=record['id'])
        if on is not None:
            record['session'] = _session_()
            _journal_(on, 'submit:BEGIN', note=json.dumps(record, indent=1, default=str), **where)
        try:
            self._create_(record)
        except Exception as e:
            if on is not None:
                _journal_(on, 'submit:ERROR', inline_note=True,
                          note=f"{self._target_.MODE}:{self._target_.name} {record.get('id')} not launched: {e}",
                          **where)
            raise
        return self.RUN(self._target_, record['id'], record, on=on)

    def _journaled_(self, adopted: bool) -> list:
        """The runs in scope, oldest first, each as `(Run, its submit:BEGIN row)`; a Run's `closing` is `(event, datetime)` or None.

        A run's `submit:*` entries are in the `PipeJournal`, on the pipeline that shipped it; those
        from before were on its entry, in the entry's own journal, and are read there too. A
        pipeline's runs are its own; a datablock's are those whose Job wrote its `build:*` entries
        (their `run` column), and those of the pipelines it is the entry of; a class's or an
        anchor's, the same for every datablock of it; a pipeline class's, every pipeline's of it.
        """
        if self._what_ is None:
            return []
        rows = self._submit_rows_(adopted)
        if rows.empty:
            return []
        if self._target_ is not None:
            rows = rows[rows['cluster'].isna() | (rows['cluster'] == self._target_.name)]
        rows = rows.sort_values('datetime')
        begins, closings = rows[rows['event'] == 'submit:BEGIN'], rows[rows['event'] != 'submit:BEGIN']
        out = []
        for _, row in begins.drop_duplicates('id').iterrows():
            with open(row['note']) as f:
                record = json.load(f)
            if self._target_ is not None and record.get('target') != self._target_.name:
                continue
            t = _rebuild_(record)
            if self._pipeline_ is not None and row['origin'] == 'pipe' and row['hash'] == self._pipeline_.hash:
                own = self._pipeline_
            elif self._dblk_ is not None and row['origin'] == 'entry' and row['hash'] == self._dblk_.hash:
                own = self._dblk_
            else:
                own = None
            run = t.RUNNER.RUN(t, record['id'], record, on=own)
            if own is None:  # Found otherwise: what to journal its close on, when it comes to that.
                run._journal_on_ = functools.partial(_journal_on_of_, record, row['origin'], row['hash'])
            closed = [(e, d) for e, d, r, n in zip(closings['event'], closings['datetime'], closings['run'],
                                                    closings['note'])
                      if r == record['id'] or (not isinstance(r, str) and _names_(str(n), record['id']))]
            run.closing = closed[-1] if closed else None
            run._closed_ = bool(closed)
            run.entrypath = row['entrypath']
            out.append((run, row))
        return out

    def _submit_rows_(self, adopted: bool) -> pd.DataFrame:
        """Every `submit:*` row in scope, from the `PipeJournal` (`origin` 'pipe') and from entries' own journals ('entry'), with `entrypath` and `run`."""
        from .pipelines import Pipeline  # pipelines imports this module
        datalake = self._datalake_ or default_datalake()
        if self._pipeline_ is not None:
            pipe = _read_(PipeJournal, f"{PipeJournal.ANCHOR}/{self._pipeline_.anchor}", datalake, event='^submit:')
            pipe = pipe[pipe['hash'] == self._pipeline_.hash] if not pipe.empty else pipe
            dblk = self._dblk_
            anchor = _read_(BuildJournal, dblk.anchor, datalake)
            mine = _of_dblk_(anchor, dblk, adopted) if not anchor.empty else None
            if adopted and mine is not None:  # The runs that built what it adopted: their build entries name them.
                built = anchor[mine & (anchor['hash'] != dblk.hash)]
                pipe = pd.concat([pipe, _submits_of_runs_(set(built['run'].dropna()), datalake)])
            entry = anchor[mine & anchor['event'].astype(str).str.startswith('submit:')] if mine is not None else anchor
        elif isinstance(self._what_, type) and issubclass(self._what_, Pipeline):  # Every pipeline of the class.
            pipe = _read_(PipeJournal, f"{PipeJournal.ANCHOR}/{self._what_.anchor}", datalake, event='^submit:')
            entry = pd.DataFrame()
        else:  # A datablock, a Datablock class or an anchor.
            anchor = _read_(BuildJournal, self._anchor_(), datalake)
            if not anchor.empty and self._dblk_ is not None:
                anchor = anchor[_of_dblk_(anchor, self._dblk_, adopted)]
            entry = anchor[anchor['event'].astype(str).str.startswith('submit:')] if not anchor.empty else anchor
            # The pipelines it is the entry of: the datablock's, or for a class any under its anchor.
            entrypaths = ({self._dblk_.anchorkeypath} | (set(anchor['anchorkeypath'].dropna()) if not anchor.empty else set())
                          if self._dblk_ is not None else f"/{self._anchor_()}/")
            pipe = _submits_of_runs_(set(anchor['run'].dropna()) if not anchor.empty else set(), datalake,
                                     entrypaths=entrypaths)
        pipe = pipe.assign(origin='pipe') if not pipe.empty else pipe
        if not entry.empty:
            entry = entry.assign(origin='entry', entrypath=entry['anchorkeypath'])
        rows = pd.concat([f for f in (pipe, entry) if not f.empty], ignore_index=True) if not (
            pipe.empty and entry.empty) else pd.DataFrame()
        for c in ('cluster', 'run', 'entrypath', 'origin'):
            if not rows.empty and c not in rows.columns:
                rows[c] = None
        return rows

    def _anchor_(self) -> str:
        """What the journal is read by: the datablock's anchor, or the class's, or the anchor given."""
        if self._dblk_ is not None:
            return self._dblk_.anchor
        return self._what_.anchor if isinstance(self._what_, type) else self._what_

    def _check_(self, options: dict) -> None:
        pass

    def _check_options_(self) -> dict:
        """The options a `dry_run='cluster'` run takes over the call's: the least the kind can run on."""
        return {}

    def _prepare_(self, command: str, **options) -> dict:
        raise NotImplementedError

    def _dry_run_(self, record: dict, dry_run) -> None:
        pass

    def _stage_(self, record: dict, **options) -> dict:
        return {}

    def _create_(self, record: dict) -> None:
        raise NotImplementedError


class Submit:
    """What `Pipeline.submit` hands out, kept for the code and commands that call it: a `Runner` on which any method of the entry is shipped.

    `p.submit(t).build_tree()` and `p.submit(t).entry.build()` are `p.runner(t).ship(...)`; `runs`,
    `cancel`, `rundash` and `ship` are the Runner's. New code calls `Pipeline.runner`.
    """

    def __init__(self, runner: Runner):
        self._runner_ = runner

    def __repr__(self):
        return f"{type(self).__name__}({self._runner_!r})"

    def __getattr__(self, method: str):
        if method.startswith('_'):
            raise AttributeError(method)
        if method in ('entry', 'entrypoint'):
            return self
        if method in ('ship', 'runs', 'cancel', 'rundash', 'pipeline', 'target'):
            return getattr(self._runner_, method)
        entry = self._runner_.pipeline.entry
        if not callable(getattr(type(entry), method, None)):
            raise AttributeError(f"{type(entry).__name__} has no method {method!r} to submit")
        return functools.partial(self._runner_.ship, method)


def clusters(mode: str | None = None) -> dict:
    """The clusters the installed packages name as `dbx.clusters` entry points, by name -- of kind *mode* only, when given.

    Entry points of the group's old name, `dbx.targets`, count too, with a DeprecationWarning; a name
    in both is `dbx.clusters`'.
    """
    old = metadata.entry_points(group=_OLD_GROUP_)
    if old:
        warnings.warn(f"entry points of group {_OLD_GROUP_!r} ({', '.join(sorted(ep.name for ep in old))}): rename "
                      f"the group {GROUP!r} in the package's pyproject.toml, and reinstall it", DeprecationWarning,
                      stacklevel=2)
    found = [t for group in (_OLD_GROUP_, GROUP) for ep in metadata.entry_points(group=group)
             for t in _flatten_(ep.load())]
    return {t.name: t for t in found if mode is None or t.MODE == mode}


def cluster(t: 'str | Target', mode: str | None = None) -> Target:
    """*t* when a Target, else the one of that name among `clusters()`; of kind *mode* when given."""
    if isinstance(t, str):
        named = clusters()
        if t not in named:
            raise KeyError(f"No cluster {t!r} among the {GROUP!r} entry points' {sorted(named)}: pass the cluster "
                           f"itself (e.g. a dbx.kube.Cluster), or name it in a package's {GROUP!r} entry point")
        t = named[t]
    if mode is not None and t.MODE != mode:
        raise ValueError(f"Target {t.name!r} is {t.MODE!r}, not {mode!r}")
    return t


def runner(what, target: 'str | Target | None' = None, *, mode: str | None = None, force: bool = False,
           datalake: str | None = None, **options) -> Runner:
    """The `Runner` of *what* on *target* -- of the target's kind, or a plain one with no target.

    *what* is a Pipeline (its entry's runs, and what launches them: `Pipeline.runner`), a datablock
    (its runs), or a Datablock class or anchor (the runs of every datablock of it, whatever its
    hash), whose journal is read in *datalake* -- by default `DBX_DATALAKE`:

        dbx.runner('sctx.datablocks.SCQwenVLContextModelEval').rundash()
        dbx.runner(p, 'crusoe').cancel()
    """
    t = _target_of_(target, mode) if target is not None else None
    return (t.RUNNER if t is not None else Runner)(what, t, force=force, datalake=datalake, **options)


def runs(what, target: 'str | Target | None' = None, *, open: bool | None = None, adopted: bool = False,
         datalake: str | None = None) -> list:
    """`runner(what, target, datalake=datalake).runs(open=open, adopted=adopted)`: the runs of a pipeline, a datablock, or a class or anchor."""
    return runner(what, target, datalake=datalake).runs(open=open, adopted=adopted)


def rundash(what, target: 'str | Target | None' = None, *, adopted: bool = False, live: bool = True,
            datalake: str | None = None) -> pd.DataFrame:
    """`runner(what, target, datalake=datalake).rundash(...)`: the runs of a pipeline, a datablock, or a class or anchor, as a table."""
    return runner(what, target, datalake=datalake).rundash(adopted=adopted, live=live)


def verify(block, hash: str) -> dict:
    """What a `dry_run='cluster'` run reports: *block*, evaluated there, has *hash*; where it is, and whether it is valid there.

    Raises ValueError when the hash differs -- code or configuration there that is not here.
    """
    if block.hash != hash:
        raise ValueError(f"{type(block).__name__} evaluates to hash {block.hash} here, not {hash} as where it "
                         f"was submitted from: the code or its configuration differs")
    return dict(hash=block.hash, anchorkeypath=block.anchorkeypath, valid=block.valid(), dbx=os.path.dirname(os.path.abspath(__file__)))


# Helpers ------------------------------------------------------------------------------------------- #

# A closed run's status, by its closing event.
_CLOSED_STATUS_ = {'submit:END': 'complete', 'submit:ERROR': 'failed', 'submit:CANCEL': 'cancelled'}


def _target_of_(t, mode=None) -> Target:
    # For Runner and runner, whose argument named target is where the run goes.
    return cluster(t, mode)


def _of_dblk_(j: pd.DataFrame, dblk, adopted: bool) -> pd.Series:
    """The rows of journal frame *j* that are *dblk*'s: of its hash, and with *adopted* of the builds its redirection reads from.

    A build adopted through a specialization is where the redirection's paths are: a row whose
    `anchorkeypath` one of them is under. One of another class is in that class's journal, not *j*.
    """
    mask = j['hash'] == dblk.hash
    paths = (getattr(dblk, '_redirected_paths_', None) or {}) if adopted else {}
    sources = {a for a in j['anchorkeypath'].dropna().unique()
               if any(str(p).startswith(f"{a.rstrip('/')}/") for p in paths.values())}
    return mask | j['anchorkeypath'].isin(sources)


def _journal_(on, event: str, **kwargs) -> str:
    """Journal *event* on *on*: a pipeline in the `PipeJournal` (with its entrypath), a datablock in its own journal."""
    from .pipelines import Pipeline  # pipelines imports this module
    if isinstance(on, Pipeline):
        return PipeJournal.write(on, event, entrypath=on.entrypath, **kwargs)
    return on.write_journal_entry(event=event, **kwargs)


def _journal_on_of_(record: dict, origin: str, hash: str):
    """What a run's record was journaled on, again: its expression's pipeline quote evaluated -- the pipeline (*origin* 'pipe') or its entry ('entry') -- None when it has no quote, or that is not of *hash* now (the code changed)."""
    for node in ast.walk(ast.parse(record.get('expression') or 'None', mode='eval')):
        if isinstance(node, ast.Call) and getattr(node.func, 'attr', None) == 'eval' and node.args \
                and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            pipeline = databits.eval(node.args[0].value)
            on = pipeline if origin == 'pipe' else pipeline.entry
            return on if on.hash == hash else None
    return None


def _read_(journal, anchor, datalake, **filters) -> pd.DataFrame:
    """*journal*'s entries under *anchor* in *datalake*, filtered, as a plain frame; empty when there are none."""
    try:
        j = journal.read(anchor, datalake=datalake, index=None, **filters)
    except FileNotFoundError:
        return pd.DataFrame()
    j = pd.DataFrame(j) if j is not None else pd.DataFrame()
    for c in ('cluster', 'run'):
        if not j.empty and c not in j.columns:
            j[c] = None
    return j


def _submits_of_runs_(runs: set, datalake, *, entrypaths: 'set | str' = frozenset()) -> pd.DataFrame:
    """The `PipeJournal`'s `submit:*` entries of *runs*, or of pipelines whose entry is one of *entrypaths* -- or, a str, is under that anchor (`/<anchor>/`)."""
    if not runs and not entrypaths:
        return pd.DataFrame()
    j = _read_(PipeJournal, PipeJournal.ANCHOR, datalake, event='^submit:')
    if j.empty:
        return j
    entries = (j['entrypath'].astype(str).str.contains(entrypaths, regex=False) if isinstance(entrypaths, str)
               else j['entrypath'].isin(entrypaths))
    return j[j['run'].isin(runs) | entries]


def _names_(note: str, run_id: str) -> bool:
    """Whether a closing entry's *note* -- written before entries named their run -- is about run *run_id*."""
    return re.search(rf"(^|[\s/]){re.escape(run_id)}(\s|$)", note) is not None


def _live_status_(run: Run) -> str:
    """*run*'s status as its cluster has it; `gone` when the cluster no longer has it, `unknown: ...` when it cannot be asked."""
    try:
        return run.status()
    except Exception as e:
        return 'gone' if 'NotFound' in str(e) or 'not found' in str(e) else f"unknown: {str(e).splitlines()[0][:80]}"


def _round_(td) -> pd.Timedelta:
    return pd.Timedelta(seconds=round(td.total_seconds()))


def _flatten_(obj) -> list:
    """The Targets in an entry point's object: a Target, or a dict or iterable of them."""
    if isinstance(obj, Target):
        return [obj]
    return [t for t in (obj.values() if isinstance(obj, dict) else obj) if isinstance(t, Target)]


def _spec_(t: Target) -> dict | None:
    """*t* as JSON -- its class and its fields -- when a dataclass; what `_rebuild_` makes it again from."""
    if not dataclasses.is_dataclass(t):
        return None
    return {'class': f"{type(t).__module__}:{type(t).__qualname__}",
            'fields': {f.name: getattr(t, f.name) for f in dataclasses.fields(t) if f.init}}


def _rebuild_(record: dict) -> Target:
    """The target a journaled record ran on: from its `target_spec`, else (an older record) by its name."""
    spec = record.get('target_spec')
    if not spec:
        return cluster(record['target'])
    module, qualname = spec['class'].split(':')
    cls = functools.reduce(getattr, qualname.split('.'), importlib.import_module(module))
    fields = {f.name: f for f in dataclasses.fields(cls)}
    kw = {k: tuple(v) if isinstance(v, list) and isinstance(fields[k].default, tuple) else v
          for k, v in spec['fields'].items() if k in fields}
    return cls(**kw)


def _literal_(value) -> bool:
    """Whether *value* survives `repr` and `ast.literal_eval`: what a shipped call's arguments must."""
    try:
        return ast.literal_eval(repr(value)) == value
    except (ValueError, SyntaxError):
        return False


def _session_() -> str:
    return str(BuildJournal.current().session)
