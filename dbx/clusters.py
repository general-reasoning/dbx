"""Running on a cluster: a pipeline's call, or a shell command, run somewhere else, and journaled here.

A run goes to a *cluster*: a Kubernetes cluster with Kueue (`dbx.kube.KubeCluster`), and
later a Slurm partition of one (`SlurmCluster`); every kind subclasses `Cluster`. It is a value, passed where it is
used -- nothing is registered:

    LAMBDA = dbx.kube.KubeCluster('lambda', namespace=..., queue=..., ...)
    my_pipeline('preset').runner(LAMBDA).build_tree()

A name works too, for a `dbx.pprint` one-liner, a console script or a new shell: `cluster('lambda')`
looks it up, each time, among the clusters the installed packages name as entry points of group
`dbx.clusters` -- each a Cluster, or a dict or list of them -- imported for the lookup and kept nowhere:

    [project.entry-points."dbx.clusters"]
    myproject = "myproject.clusters:CLUSTERS"

`Pipeline.runner(cluster, **options)` hands out the pipeline's runs as a `Runner` of the cluster's
kind -- `cluster.RUNNER`, e.g. `dbx.kube.KubeRunner` -- whose `build_tree()` (or `launch('<method>',
...)`) runs there: `dbx.eval('<the pipeline's quote>').entry.<method>(...)`, under `dbx.pprint`,
in the pipeline's own datalake if it was given one, else the runner's `env`'s `DBX_DATALAKE`, else the cluster's. It returns a `Run`: `status`, `wait`, `logs`,
`cancel`.

A run is journaled by both sides, each in its own journal, on the pipeline, each entry with the
pipeline's `entrypath` and the run's `cluster` and `run` (its id). The submitter, in the PipeJournal
of the datalake it submitted from:

    job:submit      before it is launched; its note (a file) the whole record, as JSON: the kind and
                    the cluster itself (`cluster_spec`), the run's id, the method, the command and
                    expression, the submitting session, and what the kind adds -- a Kube Job's
                    manifest, image, revision, synced workdirs
    job:complete    when it sees it complete: `Run.wait`, or a `Runs.dash` that asks the cluster
    job:fail        when launching it fails, or it sees it fail (with its log tail)
    job:cancel      `Run.cancel`

and the run itself (`run`, what its Job evaluates), in the PipeJournal of the datalake it runs in:

    job:start       before the call
    job:end         after it: 'ok', or what it raised

The two journals are one where the cluster mounts the submitter's datalake, and two otherwise (a
cluster with a filesystem of its own): `Runs.dash` reads the run's from there, by URL. They agree
on `(session, run)`: the run joins the submitting command's session (`DBX_SESSION`), and its Job
sets `DBX_CLUSTER` and `DBX_RUN`, so every entry it writes -- its own `job:*`, each datablock's
`build:*`, its exec-journal row -- names the run. A Runner -- `Pipeline.runner`, or `runner(what)`
of a datablock, a Datablock class or anchor, or a Pipeline class -- reads them back (`runs`,
which prints as their dash, and `cancel`), each run as its Run, its cluster rebuilt from the record -- to `status`,
`logs`, `wait` on or `cancel` from another process, a later session. Runs journaled before
2026-11-04 were `submit:BEGIN`, `submit:END`, `submit:ERROR`, `submit:CANCEL` -- some on the
pipeline's entry, in its own journal -- and are read as these.

The run's own record is in the datalake it ran in (`DBX_DATALAKE` there): the call runs as
`dbx.pprint --capture-output`, under the submitting command's session, so its exec-journal row,
every build-journal entry it writes (each with its `log`) and its captured stdout and stderr are
under that session there, its row told from the submitter's by its `run`. `Run.command()` is that
row, an `ExecJournalEntry`, read from the record's datalake or the *datalake* given -- here on a
cluster that mounts it, on the cluster's own machines otherwise, or by URL (`KubeCluster.url`) -- and
`Run.output()` prints the capture: the journal is where the output is, and the one way to it.

`dry_run`, an option of every kind: True prepares the record and launches nothing; a kind may check
more without launching (`'server'`: Kube's API server validates the Job); `'cluster'` launches a
small run on the cluster that only evaluates the pipeline's quote there and `verify`s the entry --
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
from importlib import metadata

import pandas as pd

from . import databits
from .databits import default_datalake
from .journals import SESSION_ENV, BuildJournal, PipeJournal, execjournal


__all__ = ['Cluster', 'Run', 'Runs', 'Runner', 'GROUP', 'cluster', 'clusters', 'run', 'runner', 'runs', 'verify']


# The entry-point group naming the installed packages' clusters.
GROUP = 'dbx.clusters'

# A run's events. The submitter's, in the PipeJournal of the datalake it submitted from:
SUBMIT, COMPLETE, FAIL, CANCEL = 'job:submit', 'job:complete', 'job:fail', 'job:cancel'
# ... and the run's own, in the PipeJournal of the datalake it ran in (`run`):
START, END = 'job:start', 'job:end'
# The submitter's events as they were named before 2026-11-04, in the journals written then: read as these.
_JOURNALED_AS_ = {'submit:BEGIN': SUBMIT, 'submit:END': COMPLETE, 'submit:ERROR': FAIL, 'submit:CANCEL': CANCEL}
# Cluster classes as named in the records journaled before 2026-11-06: rebuilt as these.
_RECORDED_AS_ = {'dbx.kube:Cluster': 'dbx.kube:KubeCluster'}
_EVENTS_ = '^(job|submit):'


class Cluster:
    """Where a run goes. A kind subclasses it -- a dataclass, so the journal can record it -- with its
    `MODE`, its `Runner` subclass `RUNNER`, and a `name`."""
    MODE: str = None
    RUNNER: type = None
    name: str


class Run:
    """A launched run: its `status`, `wait`, `logs`, `cancel`. `id` is None when nothing was launched (dry run, or `skipped`).

    *on* is what its submission is journaled on: the pipeline that launched it (in the `PipeJournal`),
    or the datablock a bare command was given (`dbx.kube.KubeCluster.submit(..., block=)`); None for neither.

    A kind implements `_status_` -- 'pending', 'running', 'complete', 'failed' -- `_logs_` and `_cancel_`.
    """

    def __init__(self, cluster: Cluster, id: str | None, record: dict, *, on=None, skipped: str | None = None):
        self.cluster, self.id, self.record, self.on, self.skipped = cluster, id, record, on, skipped
        self._closed_ = False
        # The journaled `(event, datetime)` that closed it, or None; set by `Runner.runs`.
        self.closing = None
        # A function giving what to journal its close on, when *on* is None; set by `Runner.runs`.
        self._journal_on_ = None
        # Its `job:submit` entry's datetime; set by `Runner.runs`.
        self._submitted_ = None

    def __repr__(self):
        """Its cluster, id, method, when submitted, and whether closed -- what is journaled; the cluster is not asked."""
        if self.id is None:
            what = f"skipped: {self.skipped}" if self.skipped else 'dry run'
            return f"{type(self).__name__}({self.cluster.MODE}:{self.cluster.name}, {what})"
        state = _CLOSED_STATUS_.get(self.closing[0]) if self.closing else ('closed' if self._closed_ else 'open')
        return (f"{type(self).__name__}({self.cluster.name}, {self.id}, {self.record.get('method') or '?'}, "
                f"submitted {self.submitted:%Y-%m-%d %H:%M}, {state})")

    @property
    def submitted(self) -> pd.Timestamp:
        """When it was submitted: its record's, or its `job:submit` entry's for a record without one."""
        return pd.Timestamp(self.record.get('submitted') or self._submitted_)

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

        Journals `job:complete` or `job:fail` on what it was submitted on, once. *verbose* (by default
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
                self._close_(COMPLETE, f"{self._where_()} complete")
                return True
            if s == 'failed':
                tail = self._logs_(40)
                self._close_(FAIL, f"{self._where_()} failed\n{tail}")
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
        """Stop the run; journals `job:cancel`."""
        if self.id is not None:
            self._cancel_()
            self._close_(CANCEL, f"{self._where_()} cancelled")

    def output(self, idx: int | None = None, *, basename: str | None = None, datalake: str | None = None,
               storage_options: dict | None = None) -> None:
        """Print what the run printed, as its `command` captured it: `ExecJournalEntry.output` for *idx* and *basename*.

        Available once the run ends; `logs` is the log while it runs.
        """
        self.command(datalake=datalake, storage_options=storage_options).output(idx, basename=basename)

    def command(self, *, datalake: str | None = None, storage_options: dict | None = None):
        """The run's own exec-journal row (an `ExecJournalEntry`) -- its session's, and its `run`'s -- in the datalake it ran in, or *datalake*.

        That datalake is the record's, reached at the cluster's `url` with its `fsspec_options` when the
        cluster has a filesystem of its own. The row is written when the run ends; `output` prints what
        it captured. A record from before 2026-11-04 names the run's own session, `run_session`, instead.
        """
        session = self.record.get('run_session') or self.record.get('session')
        if not session:
            raise LookupError(f"{self!r} records no session to find its row by")
        if datalake is None:
            datalake, options = _lake_of_(self)
            storage_options = options if storage_options is None else storage_options
        where = dict(datalake=datalake, storage_options=storage_options, session=f"^{session}$")
        if not self.record.get('run_session'):
            where['run'] = f"^{re.escape(str(self.id))}$"
        if len(execjournal(**where)) == 0:
            raise LookupError(f"No command of session {session} in {where['datalake']}: not over yet (its row is "
                              f"written when it ends), or that is not the datalake it ran in")
        return execjournal(iloc=0, **where)

    def _where_(self) -> str:
        return f"{self.cluster.MODE}:{self.cluster.name} {self.id}"

    def _close_(self, event: str, note: str) -> None:
        if self.on is None and self._journal_on_ is not None and not self._closed_:
            self.on = self._journal_on_()
        if self.on is not None and not self._closed_:
            _journal_(self.on, event, note=note, inline_note=True, cluster=self.cluster.name, run=self.id)
            self.closing = (event, datetime.datetime.now().isoformat())
        self._closed_ = True

    def _status_(self) -> str:
        raise NotImplementedError

    def _logs_(self, tail: int) -> str:
        raise NotImplementedError

    def _cancel_(self) -> None:
        raise NotImplementedError


class Runs(list):
    """Runs, as `Runner.runs` gives them: a list -- `runs()[-1].cancel()` -- that prints as their `dash`, each row numbered as its index.

    The printout asks the cluster for each open run's status, as `dash` does, but writes nothing:
    a run found over is closed by `dash` or `wait`, not by being printed. Its columns are
    `COLUMNS`; `dash()` has them all, and what the run journaled where it ran.
    """
    COLUMNS = ['run', 'cluster', 'method', 'status', 'submitted', 'elapsed']

    def __getitem__(self, i):
        out = super().__getitem__(i)
        return Runs(out) if isinstance(i, slice) else out

    def __repr__(self):
        if not self:
            return 'Runs([])'
        df = self.dash(pod=False, close=False)[self.COLUMNS]
        return df.assign(submitted=df['submitted'].dt.strftime('%Y-%m-%d %H:%M'),
                         elapsed=df['elapsed'].map(_hms_)).to_string()

    def cancel(self) -> 'Runs':
        """Cancel those still pending or running (asking the cluster), each journaled `job:cancel`; those cancelled.

        A run whose Job is over, or gone, is left as it is.
        """
        out = Runs()
        for run in self:
            if not run._closed_ and _live_status_(run) in ('pending', 'running'):
                run.cancel()
                out.append(run)
        return out

    def dash(self, *, live: bool = True, pod: bool = True, close: bool = True) -> pd.DataFrame:
        """These runs as a table, row `i` being `self[i]` -- what the submitter saw, and with *pod* what the run itself journaled.

        One row per run: `run, cluster, entry, method, status, submitted, started, ended, observed,
        elapsed, outcome, event`; `entry` is the key of the pipeline's entry it built.
        - From the submitter's journal: `submitted` (`job:submit`), and for a closed run `event`
          (`job:complete`, `job:fail`, `job:cancel`) and when it was `observed`. `status` is that
          event's (`complete`, `failed`, `cancelled`); with *live*, an open run's is asked of its
          cluster (`pending`, `running`, `complete`, `failed`, `gone` once the Job is deleted), and
          a run found over is closed then and there -- its `job:complete` or `job:fail` journaled -- unless not
          *close*.
        - From the run's own journal, where it ran (*pod*: read there, by URL when the cluster has
          a filesystem of its own): `started` and `ended` (`job:start`, `job:end`) and `outcome`,
          'ok' or what it raised. Where the two disagree -- the run ended 'ok', the submitter saw it
          fail (killed on its way out) -- both are shown.
        `elapsed` is `ended - started` when the run said both, else from `submitted` to `observed`,
        or to now while open.
        """
        rows = []
        now = pd.Timestamp.now()
        ran = _pod_events_(list(self)) if pod else {}
        for run in self:
            status = _CLOSED_STATUS_.get(run.closing[0]) if run.closing else (_live_status_(run) if live else 'open')
            if close and run.closing is None and status in ('complete', 'failed'):    # Over: close it, as wait() would.
                run._close_(COMPLETE if status == 'complete' else FAIL, f"{run._where_()} {status}, seen by dash")
            submitted = run.submitted
            observed = pd.Timestamp(run.closing[1]) if run.closing else None
            started, ended, outcome = ran.get(run.id, (None, None, None))
            if started is not None and ended is not None:
                elapsed = ended - started
            else:
                until = observed if observed is not None else (now if status in ('pending', 'running', 'open') else None)
                elapsed = until - submitted if until is not None else None
            m = re.search(r"\.entry\.(\w+)\(|dbx\.cluster\.(verify)\(", run.record.get('expression') or '')
            method = run.record.get('method') or ((m.group(1) or m.group(2)) if m else None)
            rows.append(dict(run=run.id, cluster=run.cluster.name,
                             entry=os.path.basename(str(getattr(run, 'entrypath', None)).rstrip('/')),
                             method=method, status=status, submitted=submitted, started=started, ended=ended,
                             observed=observed, elapsed=_round_(elapsed) if elapsed is not None else None,
                             outcome=outcome, event=run.closing[0] if run.closing else None))
        cols = ['run', 'cluster', 'entry', 'method', 'status', 'submitted', 'started', 'ended', 'observed',
                'elapsed', 'outcome', 'event']
        return pd.DataFrame(rows, columns=cols)


class Runner:
    """The runs of a pipeline, a datablock or a Datablock class: what launches them, and what lists, watches and cancels them.

    `Pipeline.runner(cluster, **options)` hands out the pipeline's, of the cluster's kind --
    `cluster.RUNNER`, e.g. `dbx.kube.KubeRunner` -- and `runner(what, cluster)` anyone's:

        r = my_pipeline('preset').runner(LAMBDA, n_gpus=8)
        r.build_tree()                          # a Run, of dbx.eval('<the quote>').entry.build_tree() there
        r.launch('try_specialize_tree', ...)    # any other method of the entry
        r.runs()                                # its runs on LAMBDA: prints as their dash
        r.runs()[-1].cancel(), r.runs().dash()  # cancel the last row; the whole table
        r.cancel()                              # those still pending or running

    `launch(method, ...)` launches `dbx.eval('<the pipeline's quote>').entry.<method>(...)` under
    `dbx.pprint`, with `DBX_DATALAKE` the run's datalake (`_run_datalake_`): the quote rebuilds the same pipeline --
    the same hashes, the same operational kwargs -- so the run builds the very datablocks a call
    here would. The arguments must be literals (`ast.literal_eval(repr(x)) == x`): they travel as
    text. `build` and `build_tree` of an entry already valid here launch nothing -- a Run `skipped`
    -- unless `force`. Only a pipeline's Runner with a cluster launches.

    Which runs `runs` and `cancel` see: a pipeline's are those it launched (in the
    `PipeJournal`, under its hash); a datablock's those whose Job wrote its build entries, and those
    of the pipelines it is the entry of; a Datablock class's or an anchor's, those of every
    datablock of it, whatever its hash; a Pipeline class's, those of every pipeline of it. With
    *adopted*, a pipeline's and a datablock's add the runs of the builds it adopted through a
    specialization. Only those on the cluster, when there is one.

    A kind subclasses it: `_check_` refuses options it does not take, at construction; `_prepare_`
    makes the record (the run's id among it); `_stage_` gets what the run needs in place;
    `_create_` launches it; `RUN` is its Run. `dry_run` -- an option of every kind -- prepares the
    record and launches nothing; a kind may check more for `dry_run='server'`, in `_dry_run_`.
    *what* is None for a bare command, launched with `_launch_`.
    """
    RUN: type = Run

    # 1. Protocol ---------------------------------------------------------------------------- #

    def __init__(self, what=None, cluster: 'str | Cluster | None' = None, *, mode: str | None = None,
                 force: bool = False, datalake: str | None = None, **options):
        from .pipelines import Pipeline  # pipelines imports this module
        self._what_, self._force_ = what, force
        self._cluster_ = _cluster_of_(cluster, mode) if cluster is not None else None
        self._pipeline_ = what if isinstance(what, Pipeline) else None
        self._dblk_ = what.entry if self._pipeline_ is not None else (
            what if what is not None and not isinstance(what, (str, type)) else None)
        self._datalake_ = datalake or getattr(self._dblk_, 'datalake', None)
        if self._cluster_ is not None:
            self._check_(options)
        self._options_ = options

    def __repr__(self):
        what = self._pipeline_.quote() if self._pipeline_ is not None else (
            self._dblk_.anchorkeypath if self._dblk_ is not None else self._anchor_())
        return f"{type(self).__name__}({what!r}, {self.cluster.name if self.cluster else None!r})"

    # 2. Declared API ------------------------------------------------------------------------ #

    def launch(self, method: str, *args, **kwargs) -> Run:
        """Launch `entry.<method>(*args, **kwargs)` on the cluster; its Run."""
        if self._pipeline_ is None or self._cluster_ is None:
            raise TypeError(f"{self!r} launches nothing: only a pipeline's Runner with a cluster does "
                            f"(`Pipeline.runner(cluster)`)")
        entry = self._pipeline_.entry
        if not callable(getattr(type(entry), method, None)):
            raise AttributeError(f"{type(entry).__name__} has no method {method!r} to ship")
        bad = [repr(a) for a in (*args, *kwargs.values()) if not _literal_(a)]
        if bad:
            raise TypeError(f"{type(self).__name__}.launch({method!r}): {', '.join(bad)} would not survive the trip "
                            f"as text; pass literals (str, numbers, None, and lists, dicts, tuples of them)")
        options = dict(self._options_)
        check = options.get('dry_run') == 'cluster'
        datalake = self._run_datalake_(options.get('env') or {})
        if method in ('build', 'build_tree') and not self._force_ and not check \
                and datalake == self._pipeline_.datalake and entry.valid():
            return self.RUN(self._cluster_, None, {}, on=self._pipeline_, skipped=f"{entry.anchorkeypath} is valid here")
        shipped = self._pipeline_.quote()
        call = ''.join([f", {a!r}" for a in args] + [f", {k}={v!r}" for k, v in kwargs.items()])
        expression = f"dbx.clusters.run(dbx.eval({shipped!r}), {method!r}{call})"
        options.setdefault('n_gpus', getattr(self._pipeline_, 'n_gpus', None) or 0)
        options.setdefault('name', f"{type(self._pipeline_).__name__.removesuffix('Pipeline')}-{method}")
        if check:  # The call is recorded, not made: the run evaluates the quote and verifies the entry.
            options.pop('dry_run')
            options = {**options, **self._check_options_(), 'name': f"{options['name']}-check"}
            options['env'] = {'DBX_SUBMIT_CALL': expression, **options.get('env', {})}
            expression = f"dbx.clusters.verify(dbx.eval({shipped!r}).entry, {entry.hash!r})"
            method = 'verify'
        # The run joins this command's session, in its own datalake: what it journals is found by (session, run).
        options['env'] = {'DBX_SUBMIT_EXPRESSION': expression, SESSION_ENV: _session_(), **options.get('env', {}),
                          'DBX_DATALAKE': datalake}
        return self._launch_('dbx.pprint --capture-output "$DBX_SUBMIT_EXPRESSION"', on=self._pipeline_,
                             expression=expression, method=method, **options)

    def build(self, *args, **kwargs) -> Run:
        """`launch('build', ...)`."""
        return self.launch('build', *args, **kwargs)

    def build_tree(self, *args, **kwargs) -> Run:
        """`launch('build_tree', ...)`."""
        return self.launch('build_tree', *args, **kwargs)

    def runs(self, *, open: bool | None = None, adopted: bool = False) -> 'Runs':
        """The runs in scope (see the class), oldest first, as `Runs`, from the journal; only those still open, or only those closed, by *open*.

        A run whose `job:complete`, `job:fail` or `job:cancel` is journaled is closed: its
        `wait` journals nothing more. Its cluster is rebuilt from its record. Printed, `Runs` is
        their dash, each row numbered as its index: `runs()[-1].cancel()` cancels the last row;
        `runs().dash()` is the whole table.
        """
        out = [run for run, _ in self._journaled_(adopted)]
        return Runs(out if open is None else [r for r in out if r._closed_ != open])

    def cancel(self, *, adopted: bool = False) -> list:
        """Cancel the runs in scope still pending or running (asking the cluster); the Runs cancelled.

        Each is journaled `job:cancel`. A run whose Job is over, or gone, is left as it is.
        """
        return self.runs(open=True, adopted=adopted).cancel()

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
    def cluster(self) -> 'Cluster | None':
        """The cluster this Runner launches on -- its `jobs()`, `queues()` -- or None for one listing every cluster's runs."""
        return self._cluster_

    # 4. Helpers ----------------------------------------------------------------------------- #

    def _run_datalake_(self, env: dict) -> str:
        """The run's `DBX_DATALAKE`: the most specific of the pipeline's own, the runner's `env`'s, the cluster's.

        A pipeline given `datalake=` carries it in its quote, and builds there wherever it runs: that
        is the run's. A pipeline that was not resolves `DBX_DATALAKE` where it is evaluated -- here,
        this machine's, which may mean nothing on the cluster -- so the run's is the runner's `env`'s,
        else the cluster's (it knows where its datalake is), and only else the one it resolved here.
        """
        if self._pipeline_._datalake_ is not None:
            return self._pipeline_.datalake
        cluster_env = (getattr(self._cluster_, 'options', None) or {}).get('env') or {}
        return env.get('DBX_DATALAKE') or cluster_env.get('DBX_DATALAKE') or self._pipeline_.datalake

    def _launch_(self, command: str, *, on=None, expression: str | None = None, method: str | None = None,
                 **options) -> Run:
        """Launch *command* with *options* (over those given at construction); journaled on *on* -- a pipeline or a datablock -- when given."""
        options = {**self._options_, **options}
        dry_run = options.pop('dry_run', False)
        dry_run = False if dry_run == 'cluster' else dry_run  # A check is a run: launch made it one.
        record = dict(executor=self._cluster_.MODE, cluster=self._cluster_.name, cluster_spec=_spec_(self._cluster_),
                      command=command, expression=expression, method=method, session=_session_(),
                      pipeline_anchor=getattr(on, 'anchor', None) if _is_pipeline_(on) else None,
                      submitted=datetime.datetime.now().isoformat())
        record.update(self._prepare_(command, **options))
        if dry_run:
            self._dry_run_(record, dry_run)
            return self.RUN(self._cluster_, None, record)
        record.update(self._stage_(record, **options))
        where = dict(cluster=self._cluster_.name, run=record['id'])
        if on is not None:
            _journal_(on, SUBMIT, note=json.dumps(record, indent=1, default=str), **where)
        try:
            self._create_(record)
        except Exception as e:
            if on is not None:
                _journal_(on, FAIL, inline_note=True,
                          note=f"{self._cluster_.MODE}:{self._cluster_.name} {record.get('id')} not launched: {e}",
                          **where)
            raise
        return self.RUN(self._cluster_, record['id'], record, on=on)

    def _journaled_(self, adopted: bool) -> list:
        """The runs in scope, oldest first, each as `(Run, its job:submit row)`; a Run's `closing` is `(event, datetime)` or None.

        The submitter's events are in the `PipeJournal`, on the pipeline that launched it; those from
        before 2026-11-04 were on its entry, in the entry's own journal, and are read there too, as
        are the `submit:*` names of before 2026-11-04. The run's own `job:start`/`job:end` are not
        read here: they are where it ran (`_pod_events_`). A
        pipeline's runs are its own; a datablock's are those whose Job wrote its `build:*` entries
        (their `run` column), and those of the pipelines it is the entry of; a class's or an
        anchor's, the same for every datablock of it; a pipeline class's, every pipeline's of it.
        """
        if self._what_ is None:
            return []
        rows = self._submit_rows_(adopted)
        if rows.empty:
            return []
        if self._cluster_ is not None:
            rows = rows[rows['cluster'].isna() | (rows['cluster'] == self._cluster_.name)]
        rows = rows.assign(event=rows['event'].replace(_JOURNALED_AS_))
        rows = rows[rows['event'].isin((SUBMIT, COMPLETE, FAIL, CANCEL))].sort_values('datetime')
        begins, closings = rows[rows['event'] == SUBMIT], rows[rows['event'] != SUBMIT]
        out = []
        for _, row in begins.drop_duplicates('id').iterrows():
            with open(row['note']) as f:
                record = json.load(f)
            if self._cluster_ is not None and _cluster_name_(record) != self._cluster_.name:
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
            run._submitted_ = row['datetime']
            out.append((run, row))
        return out

    def _submit_rows_(self, adopted: bool) -> pd.DataFrame:
        """Every run event row in scope, from the `PipeJournal` (`origin` 'pipe') and from entries' own journals ('entry'), with `entrypath` and `run`."""
        from .pipelines import Pipeline  # pipelines imports this module
        datalake = self._datalake_ or default_datalake()
        if self._pipeline_ is not None:
            pipe = _read_(PipeJournal, f"{PipeJournal.ANCHOR}/{self._pipeline_.anchor}", datalake, event=_EVENTS_)
            pipe = pipe[pipe['hash'] == self._pipeline_.hash] if not pipe.empty else pipe
            dblk = self._dblk_
            anchor = _read_(BuildJournal, dblk.anchor, datalake)
            mine = _of_dblk_(anchor, dblk, adopted) if not anchor.empty else None
            if adopted and mine is not None:  # The runs that built what it adopted: their build entries name them.
                built = anchor[mine & (anchor['hash'] != dblk.hash)]
                pipe = pd.concat([pipe, _submits_of_runs_(set(built['run'].dropna()), datalake)])
            entry = anchor[mine & anchor['event'].astype(str).str.match(_EVENTS_)] if mine is not None else anchor
        elif isinstance(self._what_, type) and issubclass(self._what_, Pipeline):  # Every pipeline of the class.
            pipe = _read_(PipeJournal, f"{PipeJournal.ANCHOR}/{self._what_.anchor}", datalake, event=_EVENTS_)
            entry = pd.DataFrame()
        else:  # A datablock, a Datablock class or an anchor.
            anchor = _read_(BuildJournal, self._anchor_(), datalake)
            if not anchor.empty and self._dblk_ is not None:
                anchor = anchor[_of_dblk_(anchor, self._dblk_, adopted)]
            entry = anchor[anchor['event'].astype(str).str.match(_EVENTS_)] if not anchor.empty else anchor
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


def clusters(mode: str | None = None) -> dict:
    """The clusters the installed packages name as `dbx.clusters` entry points, by name -- of kind *mode* only, when given."""
    found = [t for ep in metadata.entry_points(group=GROUP) for t in _flatten_(ep.load())]
    return {t.name: t for t in found if mode is None or t.MODE == mode}


def cluster(t: 'str | Cluster', mode: str | None = None) -> Cluster:
    """*t* when a Cluster, else the one of that name among `clusters()`; of kind *mode* when given."""
    if isinstance(t, str):
        named = clusters()
        if t not in named:
            raise KeyError(f"No cluster {t!r} among the {GROUP!r} entry points' {sorted(named)}: pass the cluster "
                           f"itself (e.g. a dbx.kube.KubeCluster), or name it in a package's {GROUP!r} entry point")
        t = named[t]
    if mode is not None and t.MODE != mode:
        raise ValueError(f"Cluster {t.name!r} is {t.MODE!r}, not {mode!r}")
    return t


def runner(what, cluster: 'str | Cluster | None' = None, *, mode: str | None = None, force: bool = False,
           datalake: str | None = None, **options) -> Runner:
    """The `Runner` of *what* on *cluster* -- of the cluster's kind, or a plain one with no cluster.

    *what* is a Pipeline (its entry's runs, and what launches them: `Pipeline.runner`), a datablock
    (its runs), or a Datablock class or anchor (the runs of every datablock of it, whatever its
    hash), whose journal is read in *datalake* -- by default `DBX_DATALAKE`:

        dbx.runner('sctx.datablocks.SCQwenVLContextModelEval').runs()
        dbx.runner(p, 'crusoe').cancel()
    """
    t = _cluster_of_(cluster, mode) if cluster is not None else None
    return (t.RUNNER if t is not None else Runner)(what, t, force=force, datalake=datalake, **options)


def runs(what, cluster: 'str | Cluster | None' = None, *, open: bool | None = None, adopted: bool = False,
         datalake: str | None = None) -> list:
    """`runner(what, cluster, datalake=datalake).runs(open=open, adopted=adopted)`: the runs of a pipeline, a datablock, or a class or anchor."""
    return runner(what, cluster, datalake=datalake).runs(open=open, adopted=adopted)


def run(pipeline, method: str, *args, **kwargs):
    """What a run evaluates where it runs: *pipeline*'s entry's *method*, between the run's own `job:start` and `job:end`.

    Both are journaled on *pipeline*, in the PipeJournal of this process's datalake -- the
    submitter's own when the cluster mounts it, another otherwise -- with the run's `cluster` and
    `run` (`DBX_CLUSTER`, `DBX_RUN`) and under the submitter's session (`DBX_SESSION`). `job:end`'s
    note is 'ok', or what was raised -- raised again.
    """
    where = dict(cluster=os.environ.get('DBX_CLUSTER'), run=os.environ.get('DBX_RUN'))
    _journal_(pipeline, START, **where)
    try:
        result = getattr(pipeline.entry, method)(*args, **kwargs)
    except BaseException as e:
        _journal_(pipeline, END, note=f"{type(e).__name__}: {e}", inline_note=True, **where)
        raise
    _journal_(pipeline, END, note='ok', inline_note=True, **where)
    return result


def verify(block, hash: str) -> dict:
    """What a `dry_run='cluster'` run reports: *block*, evaluated there, has *hash*; where it is, and whether it is valid there.

    Raises ValueError when the hash differs -- code or configuration there that is not here.
    """
    if block.hash != hash:
        raise ValueError(f"{type(block).__name__} evaluates to hash {block.hash} here, not {hash} as where it "
                         f"was submitted from: the code or its configuration differs")
    return dict(hash=block.hash, anchorkeypath=block.anchorkeypath, valid=block.valid(), dbx=os.path.dirname(os.path.abspath(__file__)))


# Helpers ------------------------------------------------------------------------------------------- #

# A closed run's status, by the submitter's closing event.
_CLOSED_STATUS_ = {COMPLETE: 'complete', FAIL: 'failed', CANCEL: 'cancelled'}


def _cluster_of_(t, mode=None) -> Cluster:
    # For Runner and runner, whose argument named cluster shadows the function.
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


def _is_pipeline_(obj) -> bool:
    from .pipelines import Pipeline  # pipelines imports this module
    return isinstance(obj, Pipeline)


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


def _pod_events_(runs: list) -> dict:
    """Each of *runs*'s own `job:start` / `job:end`, where it ran: `{run id: (started, ended, outcome)}`.

    Read from the PipeJournal of the datalake the run ran in -- its record's `datalake`, by the
    cluster's URL when the cluster has a filesystem of its own -- once per (datalake, pipeline).
    A run that journaled nothing there, or whose journal cannot be reached, is left out.
    """
    out, read = {}, {}
    for run in runs:
        lake, anchor = run.record.get('datalake'), run.record.get('pipeline_anchor')
        if not lake or not anchor or run.id is None:
            continue
        url, options = _lake_of_(run)
        key = (url, anchor)
        if key not in read:
            try:
                read[key] = _read_(PipeJournal, f"{PipeJournal.ANCHOR}/{anchor}", url, storage_options=options,
                                   event='^job:(start|end)$')
            except Exception:  # Unreachable now: say nothing, rather than fail the table.
                read[key] = pd.DataFrame()
        j = read[key]
        if j.empty:
            continue
        mine = j[j['run'] == run.id]
        start, end = mine[mine['event'] == START], mine[mine['event'] == END]
        out[run.id] = (pd.Timestamp(_dt_(start['datetime'].max())) if len(start) else None,
                       pd.Timestamp(_dt_(end['datetime'].max())) if len(end) else None,
                       str(end.sort_values('datetime')['note'].iloc[-1]) if len(end) else None)
    return out


def _dt_(value) -> str:
    """A journal's `datetime` (`2026-11-04T14-06-47.123`, its colons dashed) as ISO."""
    value = str(value)
    date, _, time_ = value.partition('T')
    return f"{date}T{time_.replace('-', ':', 2)}" if time_ else value


def _read_(journal, anchor, datalake, *, storage_options=None, **filters) -> pd.DataFrame:
    """*journal*'s entries under *anchor* in *datalake*, filtered, as a plain frame; empty when there are none."""
    try:
        j = journal.read(anchor, datalake=datalake, index=None, storage_options=storage_options, **filters)
    except FileNotFoundError:
        return pd.DataFrame()
    j = pd.DataFrame(j) if j is not None else pd.DataFrame()
    for c in ('cluster', 'run'):
        if not j.empty and c not in j.columns:
            j[c] = None
    return j


def _submits_of_runs_(runs: set, datalake, *, entrypaths: 'set | str' = frozenset()) -> pd.DataFrame:
    """The `PipeJournal`'s run events of *runs*, or of pipelines whose entry is one of *entrypaths* -- or, a str, is under that anchor (`/<anchor>/`)."""
    if not runs and not entrypaths:
        return pd.DataFrame()
    j = _read_(PipeJournal, PipeJournal.ANCHOR, datalake, event=_EVENTS_)
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


def _hms_(td) -> str:
    """*td* as `H:MM:SS`, hours past 24 too; '' for none."""
    if td is None or pd.isna(td):
        return ''
    h, rest = divmod(int(td.total_seconds()), 3600)
    return f"{h}:{rest // 60:02d}:{rest % 60:02d}"


def _round_(td) -> pd.Timedelta:
    return pd.Timedelta(seconds=round(td.total_seconds()))


def _flatten_(obj) -> list:
    """The Clusters in an entry point's object: a Cluster, or a dict or iterable of them."""
    if isinstance(obj, Cluster):
        return [obj]
    return [t for t in (obj.values() if isinstance(obj, dict) else obj) if isinstance(t, Cluster)]


def _lake_of_(run: Run) -> tuple:
    """`(url, storage_options)` of the datalake *run* ran in, from here: its record's, by the cluster's `url` when it has `storage`."""
    lake, c = run.record.get('datalake'), run.cluster
    if lake and getattr(c, 'storage', None):
        return c.url(lake), c.fsspec_options()
    return lake, None


def _cluster_name_(record: dict) -> str | None:
    """The name of the cluster a journaled record ran on: `cluster`, or `target` in a record from before 2026-11-09."""
    return record.get('cluster') or record.get('target')


def _spec_(t: Cluster) -> dict | None:
    """*t* as JSON -- its class and its fields -- when a dataclass; what `_rebuild_` makes it again from."""
    if not dataclasses.is_dataclass(t):
        return None
    return {'class': f"{type(t).__module__}:{type(t).__qualname__}",
            'fields': {f.name: getattr(t, f.name) for f in dataclasses.fields(t) if f.init}}


def _rebuild_(record: dict) -> Cluster:
    """The cluster a journaled record ran on: from its `cluster_spec`, else (an older record) by its name.

    Records journaled before 2026-11-09 name it `target` and `target_spec`: read as these.
    """
    spec = record.get('cluster_spec') or record.get('target_spec')
    if not spec:
        return cluster(_cluster_name_(record))
    module, qualname = _RECORDED_AS_.get(spec['class'], spec['class']).split(':')
    cls = functools.reduce(getattr, qualname.split('.'), importlib.import_module(module))
    fields = {f.name: f for f in dataclasses.fields(cls)}
    kw = {k: tuple(v) if isinstance(v, list) and isinstance(fields[k].default, tuple) else v
          for k, v in spec['fields'].items() if k in fields}
    return cls(**kw)


def _literal_(value) -> bool:
    """Whether *value* survives `repr` and `ast.literal_eval`: what a launched call's arguments must."""
    try:
        return ast.literal_eval(repr(value)) == value
    except (ValueError, SyntaxError):
        return False


def _session_() -> str:
    return str(BuildJournal.current().session)
