"""Submitting: a pipeline's call, or a shell command, run somewhere else, and journaled here.

A *target* is where a run goes: a Kubernetes cluster with Kueue (`dbx.kube.Cluster`), and later a
Slurm partition. A project registers its targets by name -- `register(target)`, at the import of the
module defining them, which it names in its packaging as an entry point of group `dbx.targets`, so
`target('lambda')` finds it without the project being imported first:

    [project.entry-points."dbx.targets"]
    myproject = "myproject.clusters"

`Pipeline.submit(target, **options)` hands out the pipeline's entrypoint as a `Submit` of the
target's kind -- `target.SUBMIT`, e.g. `dbx.kube.KubeSubmit` -- and a method called on it runs
there: `dbx.eval('<the pipeline's quote>').entry.<method>(...)`, under `dbx.pprint`, with the
entry's datalake as `DBX_DATALAKE`. It returns a `Run`: `status`, `wait`, `logs`, `cancel`.

The run is journaled on the entrypoint, in the submitting datalake:

    submit:BEGIN    before it is launched; its note (a file) the whole record, as JSON: the kind and
                    target, the run's id, the command and expression, the submitting session, and
                    what the kind adds -- a Kube Job's manifest, image, revision, synced workdirs
    submit:END      when `Run.wait` sees it complete
    submit:ERROR    when launching it fails, or `Run.wait` sees it fail (with its log tail)
    submit:CANCEL   `Run.cancel`

so `entry.buildjournal(event='^submit')` lists every run of a block, whatever its kind, and
`runs(block)` hands each back as its Run -- to `status`, `logs`, `wait` on or `cancel` from another
process, a later session.

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
import datetime
import functools
import json
import time
import uuid
from importlib import metadata


__all__ = ['Target', 'Run', 'Submit', 'register', 'target', 'targets', 'runs', 'verify']


class Target:
    """Where a run goes. A kind subclasses it with its `MODE`, its `Submit` subclass `SUBMIT`, and a `name`."""
    MODE: str = None
    SUBMIT: type = None
    name: str


TARGETS: dict = {}
_LOADED_ = False


def register(t: Target) -> Target:
    """Make *t* a target by its name."""
    TARGETS[t.name] = t
    return t


def _load_entry_points_():
    """Import every module a package names in group `dbx.targets`: each registers its targets."""
    global _LOADED_
    if not _LOADED_:
        _LOADED_ = True
        for ep in metadata.entry_points(group='dbx.targets'):
            ep.load()


def targets(mode: str | None = None) -> dict:
    """The registered targets, by name -- of kind *mode* only, when given -- the `dbx.targets` entry points' among them."""
    _load_entry_points_()
    return {n: t for n, t in TARGETS.items() if mode is None or t.MODE == mode}


def target(t: 'str | Target', mode: str | None = None) -> Target:
    """The target named *t* (*t* itself when a Target), which must be of kind *mode* when given."""
    if isinstance(t, str):
        if t not in TARGETS:
            _load_entry_points_()
        if t not in TARGETS:
            raise KeyError(f"No target {t!r} is registered; have {sorted(TARGETS)} -- a project registers its own, "
                           f"e.g. dbx.kube.register(dbx.kube.Cluster(...)), in a module it names as a "
                           f"'dbx.targets' entry point")
        t = TARGETS[t]
    if mode is not None and t.MODE != mode:
        raise ValueError(f"Target {t.name!r} is {t.MODE!r}, not {mode!r}")
    return t


_target_of_ = target  # For Submit, whose argument is named target.


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
        from .journals import execjournal
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
            expression = f"dbx.submits.verify(dbx.eval({self._pipeline_.quote()!r}).entry, {entry.hash!r})"
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
        record = dict(executor=self._target_.MODE, target=self._target_.name, command=command, expression=expression,
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


def verify(block, hash: str) -> dict:
    """What a `dry_run='cluster'` run reports: *block*, evaluated there, has *hash*; where it is, and whether it is valid there.

    Raises ValueError when the hash differs -- code or configuration there that is not here.
    """
    import dbx
    if block.hash != hash:
        raise ValueError(f"{type(block).__name__} evaluates to hash {block.hash} here, not {hash} as where it "
                         f"was submitted from: the code or its configuration differs")
    return dict(hash=block.hash, anchorkeypath=block.anchorkeypath, valid=block.valid(), dbx=dbx.__file__)


def runs(block) -> list:
    """The runs submitted for *block* (a pipeline: its entrypoint), oldest first, as their Runs, from its journal.

    A run whose `submit:END`, `submit:ERROR` or `submit:CANCEL` is journaled is closed: its `wait`
    journals nothing more. Each target must be registered, as when it was submitted.
    """
    from .pipelines import Pipeline
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
        t = target(record['target'])
        run = t.SUBMIT.RUN(t, record['id'], record, block=block)
        run._closed_ = any(f" {record['id']} " in c or f"/{record['id']} " in c for c in closings)
        out.append(run)
    return out


def _literal_(value) -> bool:
    """Whether *value* survives `repr` and `ast.literal_eval`: what a shipped call's arguments must."""
    try:
        return ast.literal_eval(repr(value)) == value
    except (ValueError, SyntaxError):
        return False


def _session_() -> str:
    from .journals import BuildJournal
    return str(BuildJournal.current().session)
