"""Running on a cluster: a pipeline's call, or a shell command, run somewhere else, and journaled here.

A *target* is where a run goes: a Kubernetes cluster with Kueue (`dbx.kube.Cluster`), and later a
Slurm partition. It is a value, passed where it is used -- nothing is registered:

    LAMBDA = dbx.kube.Cluster('lambda', namespace=..., queue=..., ...)
    my_pipeline('preset').submit(LAMBDA).build_tree()

A name works too, for a `dbx.pprint` one-liner, a console script or a new shell: `target('lambda')`
looks it up, each time, among the targets the installed packages name as entry points of group
`dbx.targets` -- each a Target, or a dict or list of them -- imported for the lookup and kept nowhere:

    [project.entry-points."dbx.targets"]
    myproject = "myproject.clusters:CLUSTERS"

`Pipeline.submit(target, **options)` hands out the pipeline's entrypoint as a `Submit` of the
target's kind -- `target.SUBMIT`, e.g. `dbx.kube.KubeSubmit` -- and a method called on it runs
there: `dbx.eval('<the pipeline's quote>').entry.<method>(...)`, under `dbx.pprint`, with the
entry's datalake as `DBX_DATALAKE`. It returns a `Run`: `status`, `wait`, `logs`, `cancel`.

The run is journaled on the entrypoint, in the submitting datalake:

    submit:BEGIN    before it is launched; its note (a file) the whole record, as JSON: the kind and
                    the target itself (`target_spec`), the run's id, the command and expression, the
                    submitting session, and what the kind adds -- a Kube Job's manifest, image,
                    revision, synced workdirs
    submit:END      when `Run.wait` sees it complete
    submit:ERROR    when launching it fails, or `Run.wait` sees it fail (with its log tail)
    submit:CANCEL   `Run.cancel`

so `entry.buildjournal(event='^submit')` lists every run of a block, whatever its kind, and
`runs(block)` hands each back as its Run, its target rebuilt from the record -- to `status`, `logs`,
`wait` on or `cancel` from another process, a later session.

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
import time
import uuid
from importlib import metadata

from .journals import BuildJournal, execjournal


__all__ = ['Target', 'Run', 'Submit', 'target', 'targets', 'runs', 'verify']


class Target:
    """Where a run goes. A kind subclasses it -- a dataclass, so the journal can record it -- with its
    `MODE`, its `Submit` subclass `SUBMIT`, and a `name`."""
    MODE: str = None
    SUBMIT: type = None
    name: str


class Run:
    """A launched run: its `status`, `wait`, `logs`, `cancel`. `id` is None when nothing was launched (dry run, or `skipped`).

    A kind implements `_status_` -- 'pending', 'running', 'complete', 'failed' -- `_logs_` and `_cancel_`.
    """

    def __init__(self, target: Target, id: str | None, record: dict, *, block=None, skipped: str | None = None):
        self.target, self.id, self.record, self.block, self.skipped = target, id, record, block, skipped
        self._closed_ = False

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

    def wait(self, *, poll: float = 30, timeout: float | None = None) -> bool:
        """Block until the run completes (True), or *timeout* passes (False); raise RuntimeError with its log tail if it fails.

        Journals `submit:END` or `submit:ERROR` on the submitted block, once.
        """
        if self.id is None:
            return True
        start = time.monotonic()
        while timeout is None or time.monotonic() - start < timeout:
            s = self._status_()
            if s == 'complete':
                self._close_('submit:END', f"{self._where_()} complete")
                return True
            if s == 'failed':
                tail = self._logs_(40)
                self._close_('submit:ERROR', f"{self._where_()} failed\n{tail}")
                raise RuntimeError(f"{self._where_()} failed; its log ends:\n{tail}")
            time.sleep(poll)
        return False

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
        if self.block is not None and not self._closed_:
            self.block.write_journal_entry(event=event, note=note, inline_note=True)
        self._closed_ = True

    def _status_(self) -> str:
        raise NotImplementedError

    def _logs_(self, tail: int) -> str:
        raise NotImplementedError

    def _cancel_(self) -> None:
        raise NotImplementedError


class Submit:
    """A pipeline's entrypoint, as `Pipeline.submit` hands it out: a method called on it runs on the target.

    `submit(...).<method>(*args, **kwargs)` launches `dbx.eval('<the pipeline's quote>').entry.<method>(...)`
    under `dbx.pprint`, with the entry's datalake as `DBX_DATALAKE`: the quote rebuilds the same
    pipeline -- the same hashes, the same operational kwargs -- so the run builds the very blocks a call
    here would. The arguments must be literals (`ast.literal_eval(repr(x)) == x`): they travel as text.
    `build` and `build_tree` of an entrypoint already valid here launch nothing -- a Run `skipped` --
    unless `force`. `.entry` and `.entrypoint` are the Submit itself, so `submit(...).entry.build_tree()`
    reads as the call it ships.

    A kind subclasses it: `_check_` refuses options it does not take, at `submit(...)`; `_prepare_`
    makes the record (the run's id among it); `_stage_` gets what the run needs in place;
    `_create_` launches it; `RUN` is its Run. `dry_run` -- an option of every kind -- prepares the
    record and launches nothing; a kind may check more for `dry_run='server'`, in `_dry_run_`.
    *pipeline* is None for a bare command, launched with `_launch_`.
    """
    RUN: type = Run

    def __init__(self, pipeline, target: 'str | Target', *, mode: str | None = None, force: bool = False, **options):
        self._pipeline_, self._target_, self._force_ = pipeline, _target_of_(target, mode), force
        self._check_(options)
        self._options_ = options

    def __repr__(self):
        what = self._pipeline_.quote() if self._pipeline_ is not None else None
        return f"{type(self).__name__}({what!r}, {self._target_.name!r})"

    def __getattr__(self, method: str):
        if method.startswith('_') or self._pipeline_ is None:
            raise AttributeError(method)
        if method in ('entry', 'entrypoint'):
            return self
        if not callable(getattr(type(self._pipeline_.entry), method, None)):
            raise AttributeError(f"{type(self._pipeline_.entry).__name__} has no method {method!r} to submit")
        return functools.partial(self._ship_, method)

    def _ship_(self, method: str, *args, **kwargs) -> Run:
        bad = [repr(a) for a in (*args, *kwargs.values()) if not _literal_(a)]
        if bad:
            raise TypeError(f"{type(self).__name__}.{method}(): {', '.join(bad)} would not survive the trip as text; "
                            f"pass literals (str, numbers, None, and lists, dicts, tuples of them)")
        entry = self._pipeline_.entry
        options = dict(self._options_)
        check = options.get('dry_run') == 'cluster'
        if method in ('build', 'build_tree') and not self._force_ and not check and entry.valid():
            return self.RUN(self._target_, None, {}, block=entry, skipped=f"{entry.anchorkeypath} is valid here")
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
        return self._launch_('dbx.pprint --capture-output "$DBX_SUBMIT_EXPRESSION"', block=entry,
                             expression=expression, **options)

    def _launch_(self, command: str, *, block=None, expression: str | None = None, **options) -> Run:
        """Launch *command* with *options* (over those given at construction); journaled on *block* when given."""
        options = {**self._options_, **options}
        dry_run = options.pop('dry_run', False)
        dry_run = False if dry_run == 'cluster' else dry_run  # A check is a run: _ship_ made it one.
        record = dict(executor=self._target_.MODE, target=self._target_.name, target_spec=_spec_(self._target_),
                      command=command, expression=expression,
                      run_session=(options.get('env') or {}).get('DBX_JOURNAL_SESSION'),
                      submitted=datetime.datetime.now().isoformat())
        record.update(self._prepare_(command, **options))
        if dry_run:
            self._dry_run_(record, dry_run)
            return self.RUN(self._target_, None, record)
        record.update(self._stage_(record, **options))
        if block is not None:
            record['session'] = _session_()
            block.write_journal_entry(event='submit:BEGIN', note=json.dumps(record, indent=1, default=str))
        try:
            self._create_(record)
        except Exception as e:
            if block is not None:
                block.write_journal_entry(event='submit:ERROR', inline_note=True,
                                          note=f"{self._target_.MODE}:{self._target_.name} {record.get('id')} "
                                               f"not launched: {e}")
            raise
        return self.RUN(self._target_, record['id'], record, block=block)

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


def targets(mode: str | None = None) -> dict:
    """The targets the installed packages name as `dbx.targets` entry points, by name -- of kind *mode* only, when given."""
    found = [t for ep in metadata.entry_points(group='dbx.targets') for t in _flatten_(ep.load())]
    return {t.name: t for t in found if mode is None or t.MODE == mode}


def target(t: 'str | Target', mode: str | None = None) -> Target:
    """*t* when a Target, else the one of that name among `targets()`; of kind *mode* when given."""
    if isinstance(t, str):
        named = targets()
        if t not in named:
            raise KeyError(f"No target {t!r} among the 'dbx.targets' entry points' {sorted(named)}: pass the "
                           f"target itself (e.g. a dbx.kube.Cluster), or name it in a package's 'dbx.targets' "
                           f"entry point")
        t = named[t]
    if mode is not None and t.MODE != mode:
        raise ValueError(f"Target {t.name!r} is {t.MODE!r}, not {mode!r}")
    return t


def runs(block) -> list:
    """The runs submitted for *block* (a pipeline: its entrypoint), oldest first, as their Runs, from its journal.

    A run whose `submit:END`, `submit:ERROR` or `submit:CANCEL` is journaled is closed: its `wait`
    journals nothing more. Its target is rebuilt from the record.
    """
    from .pipelines import Pipeline  # pipelines imports this module
    if isinstance(block, Pipeline):
        block = block.entry
    try:
        j = block.buildjournal(event='^submit:')
    except FileNotFoundError:
        return []
    if j is None or len(j) == 0:
        return []
    j = j.sort_values('datetime')
    closings = [str(n) for e, n in zip(j['event'], j['note']) if e != 'submit:BEGIN']
    out = []
    for note in j[j['event'] == 'submit:BEGIN']['note']:
        with open(note) as f:
            record = json.load(f)
        t = _rebuild_(record)
        run = t.SUBMIT.RUN(t, record['id'], record, block=block)
        run._closed_ = any(f" {record['id']} " in c or f"/{record['id']} " in c for c in closings)
        out.append(run)
    return out


def verify(block, hash: str) -> dict:
    """What a `dry_run='cluster'` run reports: *block*, evaluated there, has *hash*; where it is, and whether it is valid there.

    Raises ValueError when the hash differs -- code or configuration there that is not here.
    """
    if block.hash != hash:
        raise ValueError(f"{type(block).__name__} evaluates to hash {block.hash} here, not {hash} as where it "
                         f"was submitted from: the code or its configuration differs")
    return dict(hash=block.hash, anchorkeypath=block.anchorkeypath, valid=block.valid(), dbx=os.path.dirname(os.path.abspath(__file__)))


# Helpers ------------------------------------------------------------------------------------------- #

def _target_of_(t, mode=None) -> Target:
    # For Submit, whose argument named target shadows the function.
    return target(t, mode)


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
        return target(record['target'])
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
