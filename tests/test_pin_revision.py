"""
Tests for pinning dbx itself at a revision, via gitpinrepos() and remote().

gitwrkreposetup() cannot rewind dbx: dbx is already imported by the time any
revision is known, and an imported module is never refreshed by a checkout. In a
Ray worker the same thing happens even earlier -- resolving the Remote.RemoteDBX
actor class imports dbx.datablocks before RemoteDBX.__init__ runs -- so the pin
has to be in place before the worker interpreter starts. That means PYTHONPATH,
which is what remote() now sets.

These tests do not need a live Ray cluster: gitpinrepos() is pure git, and the
env-var composition in remote() is checked against a stub ray module.
"""
import os
import subprocess
import sys
import types
import weakref

import pytest

import dbx._pinshim_  # noqa: F401  -- imported here only so the tests can reach it

import dbx.datablocks as dbxmod
import dbx.databits as databits_mod
import dbx.anyscale as anyscale_mod
import dbx.journals as journals_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init_git_repo(path, *, package, commits=2):
    """Create a real git repo containing an importable *package*.

    Returns a list of commit SHAs (oldest first).
    """
    pkgdir = os.path.join(path, package)
    os.makedirs(pkgdir, exist_ok=True)
    subprocess.check_call(["git", "init"], cwd=path,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.check_call(["git", "config", "user.email", "t@t.com"], cwd=path)
    subprocess.check_call(["git", "config", "user.name", "T"], cwd=path)

    shas = []
    for i in range(commits):
        with open(os.path.join(pkgdir, "__init__.py"), "w") as f:
            f.write(f"MARKER = {i}\n")
        subprocess.check_call(["git", "add", "."], cwd=path,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.check_call(["git", "commit", "-m", f"commit {i}"], cwd=path,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        shas.append(subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=path).decode().strip())
    return shas


def _head(path):
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path).decode().strip()


@pytest.fixture
def repos(tmp_path):
    """A dbx-named repo and a project repo, each with two commits."""
    dbx_repo = str(tmp_path / "dbx")
    proj_repo = str(tmp_path / "proj")
    dbx_shas = _init_git_repo(dbx_repo, package="dbx")
    proj_shas = _init_git_repo(proj_repo, package="proj")
    return types.SimpleNamespace(
        dbx_repo=dbx_repo, proj_repo=proj_repo,
        dbx_shas=dbx_shas, proj_shas=proj_shas,
        gitrepo=f"{dbx_repo}:{proj_repo}",
        pin_root=str(tmp_path / "pins"),
    )


@pytest.fixture
def stub_ray(monkeypatch):
    """A ray stub capturing init kwargs and actor construction kwargs."""
    captured = {}

    class _ActorClass:
        def __init__(self, cls):
            self._cls = cls

        def options(self, runtime_env=None, **kwargs):
            captured['actor_env'] = runtime_env
            return self

        def remote(self, **kwargs):
            captured['actor_kwargs'] = kwargs
            # A worker as pinned as its runtime_env says: `Remote` asks it.
            env = (captured.get('actor_env') or {}).get('env_vars') or {}
            return types.SimpleNamespace(environ=types.SimpleNamespace(remote=env.get))

    ray = types.ModuleType('ray')
    ray.__version__ = '9.9.9-stub'
    ray.is_initialized = lambda: False
    ray.init = lambda **kw: captured.update(init=kw)
    ray.remote = _ActorClass
    ray.get = lambda ref: ref
    ray.actor = types.SimpleNamespace(ActorHandle=type('ActorHandle', (), {}))
    ray.runtime_env = types.SimpleNamespace(
        RuntimeEnv=type('RuntimeEnv', (), {
            'known_fields': {'env_vars', 'working_dir', 'worker_process_setup_hook'}}))
    monkeypatch.setitem(sys.modules, 'ray', ray)
    monkeypatch.delenv('DBX_RAY', raising=False)    # a shell's own must not pick the cluster
    monkeypatch.setattr(databits_mod.Ray, '_connection_', None)
    monkeypatch.setattr(databits_mod.Ray, '_remotes_', weakref.WeakSet())
    captured['ray'] = ray
    return captured


# ---------------------------------------------------------------------------
# gitpinrepos
# ---------------------------------------------------------------------------

class TestGitpinrepos:

    def test_clones_both_repos_at_requested_revisions(self, repos):
        revision = f"{repos.dbx_shas[0]}:{repos.proj_shas[0]}"
        dbx_pin, proj_pin = databits_mod.gitpinrepos(
            revision, gitrepo=repos.gitrepo, pin_root=repos.pin_root)

        assert _head(dbx_pin) == repos.dbx_shas[0]
        assert _head(proj_pin) == repos.proj_shas[0]

    def test_pin_root_is_a_pythonpath_entry(self, repos):
        """The returned path must CONTAIN the package, not be it."""
        revision = f"{repos.dbx_shas[0]}:{repos.proj_shas[0]}"
        dbx_pin, proj_pin = databits_mod.gitpinrepos(
            revision, gitrepo=repos.gitrepo, pin_root=repos.pin_root)

        assert os.path.isfile(os.path.join(dbx_pin, "dbx", "__init__.py"))
        assert os.path.isfile(os.path.join(proj_pin, "proj", "__init__.py"))

    def test_pinned_revision_content_is_the_old_content(self, repos):
        """A pin is only useful if it actually carries the old code."""
        dbx_pin, _ = databits_mod.gitpinrepos(
            f"{repos.dbx_shas[0]}:{repos.proj_shas[0]}",
            gitrepo=repos.gitrepo, pin_root=repos.pin_root)

        with open(os.path.join(dbx_pin, "dbx", "__init__.py")) as f:
            assert f.read().strip() == "MARKER = 0"

    def test_does_not_touch_sys_path(self, repos):
        """Deliberately unlike gitwrkreposetup: the pin is for a FRESH interpreter,
        and mutating this one's sys.path cannot dislodge an imported dbx anyway."""
        before = list(sys.path)
        databits_mod.gitpinrepos(f"{repos.dbx_shas[0]}:{repos.proj_shas[0]}",
                           gitrepo=repos.gitrepo, pin_root=repos.pin_root)
        assert sys.path == before

    def test_reuses_an_existing_pin(self, repos):
        revision = f"{repos.dbx_shas[0]}:{repos.proj_shas[0]}"
        first = databits_mod.gitpinrepos(revision, gitrepo=repos.gitrepo, pin_root=repos.pin_root)
        marker = os.path.join(first[0], "REUSED")
        open(marker, "w").close()

        second = databits_mod.gitpinrepos(revision, gitrepo=repos.gitrepo, pin_root=repos.pin_root)

        assert second == first
        assert os.path.isfile(marker), "existing pin was re-cloned instead of reused"

    def test_distinct_revisions_get_distinct_pins(self, repos):
        old, _ = databits_mod.gitpinrepos(f"{repos.dbx_shas[0]}:{repos.proj_shas[0]}",
                                    gitrepo=repos.gitrepo, pin_root=repos.pin_root)
        new, _ = databits_mod.gitpinrepos(f"{repos.dbx_shas[1]}:{repos.proj_shas[1]}",
                                    gitrepo=repos.gitrepo, pin_root=repos.pin_root)

        assert old != new
        assert _head(old) == repos.dbx_shas[0]
        assert _head(new) == repos.dbx_shas[1]

    def test_leaves_no_staging_directories(self, repos):
        databits_mod.gitpinrepos(f"{repos.dbx_shas[0]}:{repos.proj_shas[0]}",
                           gitrepo=repos.gitrepo, pin_root=repos.pin_root)
        assert not [d for d in os.listdir(repos.pin_root) if '.staging-' in d]

    def test_unpinned_side_clones_at_head(self, repos):
        """A one-sided revision string pins the project and takes dbx at HEAD."""
        dbx_pin, proj_pin = databits_mod.gitpinrepos(
            repos.proj_shas[0], gitrepo=repos.gitrepo, pin_root=repos.pin_root)

        assert _head(dbx_pin) == repos.dbx_shas[-1]
        assert _head(proj_pin) == repos.proj_shas[0]

    def test_default_pin_root_is_held_against_gc(self, repos):
        """A TemporaryDirectory deletes its tree when collected, which would pull
        the clones out from under a worker still importing them."""
        import gc
        dbx_pin, _ = databits_mod.gitpinrepos(
            f"{repos.dbx_shas[0]}:{repos.proj_shas[0]}", gitrepo=repos.gitrepo)
        gc.collect()
        assert os.path.isdir(dbx_pin)


# ---------------------------------------------------------------------------
# remote()
# ---------------------------------------------------------------------------

class TestRemotePinning:

    def test_revision_pins_both_repos_on_pythonpath(self, repos, stub_ray, monkeypatch):
        monkeypatch.setattr(databits_mod, 'DBX_USE_WORK_REPO', repos.gitrepo)
        revision = f"{repos.dbx_shas[0]}:{repos.proj_shas[0]}"

        databits_mod.remote(revision=revision, pin_root=repos.pin_root)

        env = stub_ray['actor_env']['env_vars']
        entries = env['PYTHONPATH'].split(os.pathsep)
        assert len(entries) == 2
        assert _head(entries[0]) == repos.dbx_shas[0]
        assert _head(entries[1]) == repos.proj_shas[0]

    def test_dbx_git_repo_points_at_the_pins(self, repos, stub_ray, monkeypatch):
        """So that anything the worker resolves later agrees with what it imported."""
        monkeypatch.setattr(databits_mod, 'DBX_USE_WORK_REPO', repos.gitrepo)
        revision = f"{repos.dbx_shas[0]}:{repos.proj_shas[0]}"

        databits_mod.remote(revision=revision, pin_root=repos.pin_root)

        env = stub_ray['actor_env']['env_vars']
        assert env['DBX_GIT_REPO'] == env['PYTHONPATH']

    def test_worker_is_not_asked_to_check_out_again(self, repos, stub_ray, monkeypatch):
        """Both repos are already pinned; a revision passed to RemoteDBX would clone
        a second time, and for dbx could not take effect at all."""
        monkeypatch.setattr(databits_mod, 'DBX_USE_WORK_REPO', repos.gitrepo)
        revision = f"{repos.dbx_shas[0]}:{repos.proj_shas[0]}"

        databits_mod.remote(revision=revision, pin_root=repos.pin_root)

        assert stub_ray['actor_kwargs'] == {'revision': None}

    def test_existing_pythonpath_is_preserved_after_the_pins(self, repos, stub_ray, monkeypatch):
        monkeypatch.setattr(databits_mod, 'DBX_USE_WORK_REPO', repos.gitrepo)
        monkeypatch.setenv('PYTHONPATH', '/somewhere/else')
        revision = f"{repos.dbx_shas[0]}:{repos.proj_shas[0]}"

        databits_mod.remote(revision=revision, pin_root=repos.pin_root)

        env = stub_ray['actor_env']['env_vars']
        assert env['PYTHONPATH'].split(os.pathsep)[-1] == '/somewhere/else'

    def test_no_revision_leaves_pythonpath_alone(self, repos, stub_ray, monkeypatch):
        """Regression guard: the unpinned path must behave exactly as before."""
        monkeypatch.setattr(databits_mod, 'DBX_USE_WORK_REPO', repos.gitrepo)
        monkeypatch.delenv('PYTHONPATH', raising=False)

        databits_mod.remote()

        env = stub_ray['actor_env']['env_vars']
        assert 'PYTHONPATH' not in env
        assert env['DBX_GIT_REPO'] == repos.gitrepo
        assert stub_ray['actor_kwargs'] == {'revision': None}


# ---------------------------------------------------------------------------
# The self-clone shim
# ---------------------------------------------------------------------------

class TestPinShim:
    """dbx._pinshim_ runs in the worker BEFORE dbx is imported, so it may use
    nothing but the standard library. Verified against a live cluster too, where
    the hook reported `dbx_in_sys_modules_at_hook_entry: False`."""

    @pytest.fixture
    def shim(self):
        import dbx._pinshim_ as shim
        return shim

    @pytest.fixture
    def clean_path(self):
        before = list(sys.path)
        yield
        sys.path[:] = before

    def test_setup_clones_at_the_revision_and_leads_sys_path(
            self, shim, repos, tmp_path, monkeypatch, clean_path):
        monkeypatch.setenv('DBX_PIN_SOURCE', f"{repos.dbx_repo} {repos.proj_repo}")
        monkeypatch.setenv('DBX_PIN_REVISION', f"{repos.dbx_shas[0]}:{repos.proj_shas[0]}")
        monkeypatch.setenv('DBX_PIN_NODE_ROOT', str(tmp_path / 'node'))

        shim.setup()

        head = sys.path[:2]
        assert any(_head(p) == repos.dbx_shas[0] for p in head)
        assert any(_head(p) == repos.proj_shas[0] for p in head)

    def test_clones_are_shared_between_worker_processes(
            self, shim, repos, tmp_path, monkeypatch, clean_path):
        """Named for the revision, so N workers on a node clone once, not N times."""
        root = str(tmp_path / 'node')
        monkeypatch.setenv('DBX_PIN_SOURCE', repos.dbx_repo)
        monkeypatch.setenv('DBX_PIN_REVISION', repos.dbx_shas[0])
        monkeypatch.setenv('DBX_PIN_NODE_ROOT', root)

        shim.setup()
        marker = os.path.join(sys.path[0], 'SHARED')
        open(marker, 'w').close()
        shim.setup()

        assert os.path.isfile(marker), "second worker re-cloned instead of reusing"

    def test_source_and_revision_counts_must_match(self, shim, repos, monkeypatch, clean_path):
        monkeypatch.setenv('DBX_PIN_SOURCE', repos.dbx_repo)
        monkeypatch.setenv('DBX_PIN_REVISION', 'aaa:bbb')
        with pytest.raises(RuntimeError, match='1 sources but 2 revisions'):
            shim.setup()

    def test_pin_paths_go_first_as_they_are(self, shim, tmp_path, monkeypatch, clean_path):
        """`pin_mode='pin_root'`'s clones, ahead of whatever Ray put first -- the driver's directory."""
        a, b = tmp_path / 'dbx-pin', tmp_path / 'proj-pin'
        a.mkdir(), b.mkdir()
        monkeypatch.delenv('DBX_PIN_SOURCE', raising=False)
        monkeypatch.delenv('DBX_PIN_REVISION', raising=False)
        monkeypatch.setenv('DBX_PIN_PATHS', os.pathsep.join([str(a), str(b)]))
        shim.setup()
        assert sys.path[:2] == [str(a), str(b)]

    def test_a_pin_path_not_on_the_node_raises(self, shim, tmp_path, monkeypatch, clean_path):
        monkeypatch.setenv('DBX_PIN_PATHS', str(tmp_path / 'elsewhere'))
        with pytest.raises(RuntimeError, match='not on this node'):
            shim.setup()

    def test_no_env_is_a_no_op(self, shim, monkeypatch, clean_path):
        monkeypatch.delenv('DBX_PIN_SOURCE', raising=False)
        monkeypatch.delenv('DBX_PIN_REVISION', raising=False)
        before = list(sys.path)
        shim.setup()
        assert sys.path == before

    def test_a_bad_revision_raises_rather_than_running_unpinned(
            self, shim, repos, tmp_path, monkeypatch, clean_path):
        """A worker that quietly came up unpinned would return wrong hashes."""
        monkeypatch.setenv('DBX_PIN_SOURCE', repos.dbx_repo)
        monkeypatch.setenv('DBX_PIN_REVISION', 'not-a-real-sha')
        monkeypatch.setenv('DBX_PIN_NODE_ROOT', str(tmp_path / 'node'))
        with pytest.raises(subprocess.CalledProcessError):
            shim.setup()
        assert not [d for d in os.listdir(tmp_path / 'node') if '.staging-' in d]


class TestRemotePinMode:

    @pytest.fixture(autouse=True)
    def _repo_globals(self, repos, monkeypatch):
        monkeypatch.setattr(databits_mod, 'DBX_USE_WORK_REPO', repos.gitrepo)
        monkeypatch.setattr(databits_mod, '_DBX_GIT_REPO_', repos.gitrepo)

    def _env(self, stub_ray):
        return stub_ray['actor_env']

    def test_shim_ships_the_bootstrap_and_names_the_hook(self, repos, stub_ray):
        rev = f"{repos.dbx_shas[0]}:{repos.proj_shas[0]}"
        databits_mod.remote(revision=rev, pin_mode='shim')
        renv = self._env(stub_ray)
        assert renv['env_vars'][databits_mod.RAY_SETUP_HOOK_ENV_VAR] == 'dbxpinshim.setup'
        (shim,) = stub_ray['init']['runtime_env']['py_modules']
        assert os.path.basename(shim) == 'dbxpinshim.py' and os.path.isfile(shim), "shipped as the job's py_modules"
        assert renv['env_vars']['DBX_PIN_SOURCE'] == f"{repos.dbx_repo} {repos.proj_repo}"
        assert renv['env_vars']['DBX_PIN_REVISION'] == rev
        assert 'PYTHONPATH' not in renv['env_vars'], "shim pins by sys.path, not PYTHONPATH"

    def test_pin_root_puts_its_paths_first_through_the_hook(self, repos, stub_ray):
        """PYTHONPATH comes after the driver's directory Ray puts first; the hook puts the pins ahead of it."""
        rev = f"{repos.dbx_shas[0]}:{repos.proj_shas[0]}"
        databits_mod.remote(revision=rev, pin_mode='pin_root', pin_root=repos.pin_root)
        env = self._env(stub_ray)['env_vars']
        assert env[databits_mod.RAY_SETUP_HOOK_ENV_VAR] == 'dbxpinshim.setup' and 'DBX_PIN_SOURCE' not in env, "no clone of its own"
        assert env['DBX_PIN_PATHS'].split(os.pathsep) == env['PYTHONPATH'].split(os.pathsep)[:2]

    def test_auto_takes_an_explicit_pin_root(self, repos, stub_ray):
        rev = f"{repos.dbx_shas[0]}:{repos.proj_shas[0]}"
        databits_mod.remote(revision=rev, pin_root=repos.pin_root)
        env = self._env(stub_ray)['env_vars']
        assert 'DBX_PIN_SOURCE' not in env and 'DBX_PIN_PATHS' in env

    def test_auto_prefers_shim_on_this_host(self, repos, stub_ray, monkeypatch):
        """A Ray on this host may still put the worker on another node: the head of an
        Anyscale workspace offers Ray no CPU. A temporary pin_root here is not there."""
        monkeypatch.setattr(databits_mod, 'DBX_USE_WORK_REPO', '/tmp/node-local-work-repo')
        rev = f"{repos.dbx_shas[0]}:{repos.proj_shas[0]}"
        databits_mod.remote(revision=rev)
        renv = self._env(stub_ray)
        assert renv['env_vars'][databits_mod.RAY_SETUP_HOOK_ENV_VAR] == 'dbxpinshim.setup'
        assert renv['env_vars']['DBX_PIN_SOURCE'] == f"{repos.dbx_repo} {repos.proj_repo}", \
            "cloned from the original repos, not the node-local work repo"

    def test_auto_prefers_shim_off_host(self, repos, stub_ray):
        rev = f"{repos.dbx_shas[0]}:{repos.proj_shas[0]}"
        databits_mod.remote(revision=rev, address='ray://head:10001')
        assert self._env(stub_ray)['env_vars'][databits_mod.RAY_SETUP_HOOK_ENV_VAR] == 'dbxpinshim.setup'

    def test_auto_falls_back_to_pin_root_without_the_hook(self, repos, stub_ray, monkeypatch):
        monkeypatch.setattr(stub_ray['ray'].runtime_env.RuntimeEnv, 'known_fields', {'env_vars'})
        rev = f"{repos.dbx_shas[0]}:{repos.proj_shas[0]}"
        databits_mod.remote(revision=rev, address='ray://head:10001', pin_root=repos.pin_root)
        assert databits_mod.RAY_SETUP_HOOK_ENV_VAR not in self._env(stub_ray)['env_vars']

    def test_dbx_ray_is_the_default_address(self, repos, stub_ray, monkeypatch):
        monkeypatch.setenv('DBX_RAY', 'auto')
        rev = f"{repos.dbx_shas[0]}:{repos.proj_shas[0]}"
        databits_mod.remote(revision=rev)
        assert stub_ray['init']['address'] == 'auto'
        assert self._env(stub_ray)['env_vars'][databits_mod.RAY_SETUP_HOOK_ENV_VAR] == 'dbxpinshim.setup'

    def test_an_address_passed_overrides_dbx_ray(self, repos, stub_ray, monkeypatch):
        monkeypatch.setenv('DBX_RAY', 'auto')
        databits_mod.remote(address='ray://head:10001')
        assert stub_ray['init']['address'] == 'ray://head:10001'

    def test_without_dbx_ray_ray_init_gets_no_address(self, repos, stub_ray):
        databits_mod.remote()
        assert 'address' not in stub_ray['init']

    def test_explicit_shim_without_the_hook_is_an_error(self, repos, stub_ray, monkeypatch):
        monkeypatch.setattr(stub_ray['ray'].runtime_env.RuntimeEnv, 'known_fields', {'env_vars'})
        rev = f"{repos.dbx_shas[0]}:{repos.proj_shas[0]}"
        with pytest.raises(ValueError, match='worker_process_setup_hook'):
            databits_mod.remote(revision=rev, pin_mode='shim')

    def test_a_git_url_source_implies_shim(self, repos, stub_ray):
        rev = f"{repos.dbx_shas[0]}:{repos.proj_shas[0]}"
        databits_mod.remote(revision=rev, pin_source='https://h/dbx.git https://h/proj.git')
        renv = self._env(stub_ray)
        assert renv['env_vars'][databits_mod.RAY_SETUP_HOOK_ENV_VAR] == 'dbxpinshim.setup'
        assert renv['env_vars']['DBX_PIN_SOURCE'] == 'https://h/dbx.git https://h/proj.git'

    def test_pin_source_must_match_the_revision_arity(self, repos, stub_ray):
        with pytest.raises(ValueError, match='zipped together'):
            databits_mod.remote(revision='a:b', pin_source='https://h/only-one.git')

    def test_shim_goes_with_an_explicit_working_dir(self, repos, stub_ray, tmp_path):
        """The shim is the job's py_modules, not its working_dir: the two are independent."""
        rev = f"{repos.dbx_shas[0]}:{repos.proj_shas[0]}"
        databits_mod.remote(revision=rev, pin_mode='shim', working_dir=str(tmp_path))
        job = stub_ray['init']['runtime_env']
        assert job['working_dir'] == str(tmp_path) and job['py_modules'][0].endswith('dbxpinshim.py')

    def test_pinned_revision_is_advertised_to_the_worker(self, repos, stub_ray):
        rev = f"{repos.dbx_shas[0]}:{repos.proj_shas[0]}"
        databits_mod.remote(revision=rev, pin_root=repos.pin_root)
        assert self._env(stub_ray)['env_vars']['DBX_PINNED_REVISION'] == rev

    def test_gitrepo_overrides_the_clone_source(self, repos, stub_ray):
        rev = f"{repos.dbx_shas[0]}:{repos.proj_shas[0]}"
        databits_mod.remote(revision=rev, pin_mode='shim', gitrepo=repos.gitrepo)
        assert self._env(stub_ray)['env_vars']['DBX_PIN_SOURCE'] == \
            f"{repos.dbx_repo} {repos.proj_repo}"


# ---------------------------------------------------------------------------
# A pinned interpreter refuses to re-pin
# ---------------------------------------------------------------------------

class TestPinnedGuard:

    def test_matching_revision_is_a_no_op(self, monkeypatch, repos):
        monkeypatch.setenv('DBX_PINNED_REVISION', 'dbxsha:projsha')
        monkeypatch.setattr(databits_mod, 'DBX_USE_WORK_REPO', None)
        monkeypatch.setattr(databits_mod, 'DBX_GIT_REPO', repos.gitrepo)
        databits_mod.gitwrkreposetup(revision='dbxsha:projsha')
        assert databits_mod.DBX_USE_WORK_REPO is None, "pinned interpreter should not clone"

    def test_conflicting_revision_is_refused_not_half_applied(self, monkeypatch, repos):
        """Rewinding the project under a dbx that cannot follow is how a wrong
        hash gets produced confidently."""
        warnings = []
        monkeypatch.setenv('DBX_PINNED_REVISION', 'dbxsha:projsha')
        monkeypatch.setattr(databits_mod, 'DBX_USE_WORK_REPO', None)
        monkeypatch.setattr(databits_mod, 'DBX_GIT_REPO', repos.gitrepo)
        log = dbxmod.Datalog(name='t')
        monkeypatch.setattr(log, 'warning', lambda m, *a, **k: warnings.append(m))
        databits_mod.gitwrkreposetup(revision='other:other', log=log)
        assert databits_mod.DBX_USE_WORK_REPO is None
        assert warnings and 'pinned to dbxsha:projsha' in warnings[0]

    def test_unpinned_interpreter_is_unaffected(self, monkeypatch, repos):
        monkeypatch.delenv('DBX_PINNED_REVISION', raising=False)
        monkeypatch.setattr(databits_mod, 'DBX_USE_WORK_REPO', None)
        monkeypatch.setattr(databits_mod, 'DBX_GIT_REPO', repos.gitrepo)
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        databits_mod.gitwrkreposetup(revision=f"{repos.dbx_shas[0]}:{repos.proj_shas[0]}")
        assert databits_mod.DBX_USE_WORK_REPO is not None, "unpinned setup should still clone"


# ---------------------------------------------------------------------------
# The CLI trampoline
# ---------------------------------------------------------------------------

class TestPinTrampoline:

    @pytest.fixture
    def execve(self, monkeypatch):
        captured = {}

        class _Exec(Exception):
            pass

        def fake(path, argv, env):
            captured.update(path=path, argv=argv, env=env)
            raise _Exec()

        monkeypatch.setattr(os, 'execve', fake)
        captured['sentinel'] = _Exec
        return captured

    def test_no_flags_means_no_trampoline(self, execve, monkeypatch):
        monkeypatch.delenv('DBX_PINNED_REVISION', raising=False)
        monkeypatch.setattr(sys, 'argv', ['dbx.pprint', 'some.expr()'])
        assert databits_mod.pintrampoline() is None

    def test_phase_two_does_not_trampoline_again(self, execve, monkeypatch):
        """Without this guard, exec'ing yourself is an infinite loop."""
        monkeypatch.setenv('DBX_PINNED_REVISION', 'dbxsha:projsha')
        monkeypatch.setattr(sys, 'argv', ['dbx.pprint', '--revision=dbxsha:projsha', 'e()'])
        assert databits_mod.pintrampoline() is None

    def test_revision_flag_execs_with_pins_and_guard(self, execve, monkeypatch, repos):
        rev = f"{repos.dbx_shas[0]}:{repos.proj_shas[0]}"
        monkeypatch.delenv('DBX_PINNED_REVISION', raising=False)
        monkeypatch.setattr(databits_mod, 'DBX_GIT_REPO', repos.gitrepo)
        monkeypatch.setattr(sys, 'argv',
                            ['dbx.pprint', f'--revision={rev}', f'--pin-root={repos.pin_root}', 'e()'])

        with pytest.raises(execve['sentinel']):
            databits_mod.pintrampoline()

        assert execve['argv'][-1] == 'e()'
        assert not [a for a in execve['argv'] if a.startswith('--revision=')], \
            "pin flags must not reach the expression parser in phase 2"
        assert execve['env']['DBX_PINNED_REVISION'] == rev
        assert 'DBX_USE_WORK_REPO' not in execve['env']
        pins = execve['env']['PYTHONPATH'].split(os.pathsep)
        assert _head(pins[0]) == repos.dbx_shas[0]
        assert _head(pins[1]) == repos.proj_shas[0]

    def test_existing_pythonpath_is_kept_behind_the_pins(self, execve, monkeypatch, repos):
        rev = f"{repos.dbx_shas[0]}:{repos.proj_shas[0]}"
        monkeypatch.delenv('DBX_PINNED_REVISION', raising=False)
        monkeypatch.setenv('PYTHONPATH', '/somewhere/else')
        monkeypatch.setattr(databits_mod, 'DBX_GIT_REPO', repos.gitrepo)
        monkeypatch.setattr(sys, 'argv',
                            ['dbx.pprint', f'--revision={rev}', f'--pin-root={repos.pin_root}', 'e()'])
        with pytest.raises(execve['sentinel']):
            databits_mod.pintrampoline()
        assert execve['env']['PYTHONPATH'].split(os.pathsep)[-1] == '/somewhere/else'

    def test_pin_from_reads_the_revision_off_a_selector(self, execve, monkeypatch, repos):
        rev = f"{repos.dbx_shas[0]}:{repos.proj_shas[0]}"
        monkeypatch.delenv('DBX_PINNED_REVISION', raising=False)
        monkeypatch.setattr(databits_mod, 'DBX_GIT_REPO', repos.gitrepo)
        monkeypatch.setattr(dbxmod, '_selector_probe',
                            types.SimpleNamespace(revision=rev), raising=False)
        monkeypatch.setattr(sys, 'argv',
                            ['dbx.pprint', '--pin-from=dbx.datablocks._selector_probe',
                             f'--pin-root={repos.pin_root}', 'e()'])

        with pytest.raises(execve['sentinel']):
            databits_mod.pintrampoline()

        assert execve['env']['DBX_PINNED_REVISION'] == rev

    def test_a_selector_without_a_revision_is_an_error(self, execve, monkeypatch, repos):
        monkeypatch.delenv('DBX_PINNED_REVISION', raising=False)
        monkeypatch.setattr(dbxmod, '_selector_probe', object(), raising=False)
        monkeypatch.setattr(sys, 'argv',
                            ['dbx.pprint', '--pin-from=dbx.datablocks._selector_probe', 'e()'])
        with pytest.raises(ValueError, match='no .revision'):
            databits_mod.pintrampoline()


# ---------------------------------------------------------------------------
# inst(remote=...) dispatch
# ---------------------------------------------------------------------------

class TestInstRemoteDispatch:
    """Dispatch only -- the behaviour of a real pinned worker is covered end to
    end against a live Ray cluster, not here."""

    @pytest.fixture
    def entry(self):
        import pandas as pd
        return dbxmod.BuildJournalEntry(pd.Series({
            'anchor': 'pkg.mod.Block',
            'hash': 'a' * 64,
            'revision': 'dbxsha:projsha',
        }))

    @pytest.fixture
    def recorded(self, monkeypatch):
        calls = []
        monkeypatch.setattr(dbxmod.BuildJournalEntry, 'rinst',
                            lambda self, **kw: calls.append(kw) or 'PROXY')
        return calls

    def test_default_stays_local(self, entry, recorded, monkeypatch):
        monkeypatch.setattr(dbxmod.BuildJournalEntry, 'instantiate',
                            lambda self, **kw: 'LOCAL')
        assert entry.inst() == 'LOCAL'
        assert recorded == []

    def test_remote_true_delegates_to_rinst(self, entry, recorded):
        assert entry.inst(remote=True) == 'PROXY'
        assert recorded == [{'gitrepo': None, 'revision': 'journal_entry', 'handle': None}]

    def test_remote_kwargs_are_forwarded(self, entry, recorded):
        entry.inst(remote=True, pin_root='/shared/pins', address='ray://h:10001')
        assert recorded[0]['pin_root'] == '/shared/pins'
        assert recorded[0]['address'] == 'ray://h:10001'

    def test_an_existing_handle_is_passed_through(self, entry, recorded):
        r = databits_mod.Remote(handle=object())
        entry.inst(remote=r)
        assert recorded[0]['handle'] is r

    def test_rinst_rejects_a_handle_plus_remote_kwargs(self, entry):
        """Fails before the quote is read, so a programming error costs no I/O."""
        r = databits_mod.Remote(handle=object())
        with pytest.raises(ValueError, match='existing handle'):
            entry.rinst(handle=r, pin_root='/shared/pins')

    def test_rinst_rejects_a_handle_plus_gitrepo(self, entry):
        r = databits_mod.Remote(handle=object())
        with pytest.raises(ValueError, match='gitrepo'):
            entry.rinst(gitrepo='/some/repo', handle=r)


class TestInstRemoteEqualsRinst:
    """`.rinst(...)` must be a pure shorthand for `.inst(remote=True, ...)`.

    Enforced by comparing what actually reaches `remote` on both routes,
    so a parameter added to one and forgotten on the other fails here.
    """

    EQUIVALENT_KWARGS = [
        {},
        {'revision': 'dbxsha:projsha'},
        {'gitrepo': '/repos/dbx:/repos/proj'},
        {'pin_root': '/shared/pins'},
        {'pin_mode': 'shim', 'pin_source': 'https://h/dbx.git https://h/proj.git'},
        {'address': 'ray://head:10001', 'shared_repo': True},
        {'gitrepo': '/repos/dbx:/repos/proj', 'revision': 'a:b',
         'pin_root': '/shared/pins', 'working_dir': None},
    ]

    @pytest.fixture
    def captured(self, monkeypatch):
        calls = []

        class _Handle:
            def run(self, func):
                return 'PROXY'

        # Where BuildJournalEntry.inst looks `remote` up: dbx.journals, since the split.
        monkeypatch.setattr(journals_mod, 'remote', lambda **kw: calls.append(kw) or _Handle())
        monkeypatch.setattr(dbxmod.BuildJournalEntry, 'read',
                            lambda self, *a, **k: '$pkg.mod.Block(datalake="u")')
        return calls

    @pytest.fixture
    def entry(self):
        import pandas as pd
        return dbxmod.BuildJournalEntry(pd.Series({
            'anchor': 'pkg.mod.Block', 'hash': 'a' * 64, 'revision': 'dbxsha:projsha',
        }))

    @pytest.mark.parametrize('kwargs', EQUIVALENT_KWARGS,
                             ids=lambda k: ','.join(sorted(k)) or 'defaults')
    def test_both_routes_reach_remote_identically(self, entry, captured, kwargs):
        assert entry.inst(remote=True, **kwargs) == 'PROXY'
        assert entry.rinst(**kwargs) == 'PROXY'
        via_inst, via_rinst = captured
        assert via_inst == via_rinst

    def test_an_existing_handle_is_equivalent_too(self, entry, captured):
        """remote=<Remote> is just handle=<Remote> spelled differently."""
        seen = []

        class _Handle(databits_mod.Remote):
            def run(self, func):
                seen.append(func)
                return 'PROXY'

        r = _Handle(handle=object())
        assert entry.inst(remote=r) == 'PROXY'
        assert entry.rinst(handle=r) == 'PROXY'
        assert captured == [], "an existing handle must not spin up a new worker"
        assert len(seen) == 2


# ---------------------------------------------------------------------------
# Displacing the driver's cwd
# ---------------------------------------------------------------------------

class TestRemoteWorkingDir:
    """The job's working_dir is what every worker gets: the workspace on Anyscale, as before dbx pinned anything.

    It was once an empty directory when pinning, to displace the driver's cwd, which Ray puts on a worker's
    sys.path ahead of PYTHONPATH -- a driver run from inside the dbx checkout imported the LIVE tree despite
    correct pins. The shim now puts the pins at sys.path[0] after Ray has set it up, so a working_dir
    neither carries nor shadows them, and the workers keep the workspace.
    """

    @pytest.fixture(autouse=True)
    def _work_repo(self, repos, monkeypatch):
        monkeypatch.setattr(databits_mod, 'DBX_USE_WORK_REPO', repos.gitrepo)

    def test_off_anyscale_pinning_leaves_rays_own(self, repos, stub_ray, monkeypatch):
        monkeypatch.setattr(anyscale_mod, 'present', lambda: False)
        databits_mod.remote(revision=f"{repos.dbx_shas[0]}:{repos.proj_shas[0]}", pin_root=repos.pin_root)
        assert 'working_dir' not in stub_ray['init']['runtime_env']

    def test_on_anyscale_the_workspace_is_the_working_dir(self, repos, stub_ray, monkeypatch):
        """As `dbx.gpu.run` gives it: so a Remote's worker and a gpu task's share one connection."""
        monkeypatch.setattr(anyscale_mod, 'present', lambda: True)
        monkeypatch.setenv('ANYSCALE_WORKING_DIR', '/home/ray/default')
        databits_mod.remote(revision=f"{repos.dbx_shas[0]}:{repos.proj_shas[0]}")
        assert stub_ray['init']['runtime_env']['working_dir'] == '/home/ray/default'

    def test_explicit_working_dir_is_honored(self, stub_ray, tmp_path):
        databits_mod.remote(working_dir=str(tmp_path))
        assert stub_ray['init']['runtime_env']['working_dir'] == str(tmp_path)

    def test_working_dir_none_is_none(self, stub_ray, monkeypatch):
        monkeypatch.setattr(anyscale_mod, 'present', lambda: True)
        databits_mod.remote(working_dir=None)
        assert 'working_dir' not in stub_ray['init']['runtime_env']


# ---------------------------------------------------------------------------
# Which repo path the workers are given
# ---------------------------------------------------------------------------

class TestRemoteSharedRepo:
    """The work repo is a temp dir, so only the ORIGINAL repo path can be shared.

    This used to key off `slurm` alone, leaving a cluster reached by plain
    `address` with an unreadable node-local path.
    """

    @pytest.fixture(autouse=True)
    def _repo_globals(self, repos, monkeypatch):
        monkeypatch.setattr(databits_mod, 'DBX_USE_WORK_REPO', repos.gitrepo)
        monkeypatch.setattr(databits_mod, '_DBX_GIT_REPO_', '/original/dbx:/original/proj')

    def test_local_cluster_keeps_the_work_repo(self, stub_ray, repos):
        databits_mod.remote()
        assert stub_ray['actor_env']['env_vars']['DBX_GIT_REPO'] == repos.gitrepo

    def test_address_switches_to_the_original_repo(self, stub_ray):
        databits_mod.remote(address='ray://head:10001')
        env = stub_ray['actor_env']['env_vars']
        assert env['DBX_GIT_REPO'] == '/original/dbx:/original/proj'

    def test_address_is_passed_to_ray_init(self, stub_ray):
        databits_mod.remote(address='ray://head:10001')
        assert stub_ray['init']['address'] == 'ray://head:10001'

    def test_slurm_still_switches_to_the_original_repo(self, stub_ray):
        # cancel() because Remote.__del__ cancels the Slurm job on teardown.
        databits_mod.remote(slurm=types.SimpleNamespace(ray_address=None, cancel=lambda: None))
        env = stub_ray['actor_env']['env_vars']
        assert env['DBX_GIT_REPO'] == '/original/dbx:/original/proj'

    def test_shared_repo_true_forces_the_original_repo(self, stub_ray):
        databits_mod.remote(shared_repo=True)
        env = stub_ray['actor_env']['env_vars']
        assert env['DBX_GIT_REPO'] == '/original/dbx:/original/proj'

    def test_shared_repo_false_forces_the_work_repo(self, stub_ray, repos):
        databits_mod.remote(address='ray://head:10001', shared_repo=False)
        env = stub_ray['actor_env']['env_vars']
        assert env['DBX_GIT_REPO'] == repos.gitrepo

    def test_shared_repo_needs_an_original_repo_to_share(self, stub_ray, monkeypatch):
        monkeypatch.setattr(databits_mod, '_DBX_GIT_REPO_', None)
        with pytest.raises(ValueError, match='DBX_GIT_REPO'):
            databits_mod.remote(shared_repo=True)

    def test_pins_still_win_over_the_repo_choice(self, stub_ray, repos):
        """A pinned worker imports from the pins, so DBX_GIT_REPO must name them."""
        revision = f"{repos.dbx_shas[0]}:{repos.proj_shas[0]}"
        databits_mod.remote(revision=revision, shared_repo=False, pin_root=repos.pin_root)
        env = stub_ray['actor_env']['env_vars']
        assert env['DBX_GIT_REPO'] == env['PYTHONPATH']
        assert env['DBX_GIT_REPO'] != repos.gitrepo


# ---------------------------------------------------------------------------
# One Ray connection, every Remote's
# ---------------------------------------------------------------------------

class TestRayConnection:
    """A Remote lives on its process's Ray connection: a new one would end it.

    So `remote` connects once and every Remote after rides it, each pinned by an
    actor runtime_env of its own; only another cluster -- or a connection that
    cannot carry the shim -- is a new one.
    """

    @pytest.fixture
    def ray_(self, stub_ray, repos, monkeypatch):
        monkeypatch.setattr(databits_mod, 'DBX_USE_WORK_REPO', repos.gitrepo)
        monkeypatch.setattr(databits_mod, '_DBX_GIT_REPO_', repos.gitrepo)
        ray = stub_ray['ray']
        state = {'on': False, 'inits': 0, 'shutdowns': 0, 'envs': []}

        def init(**kw):
            stub_ray['init'] = kw
            state.update(on=True, inits=state['inits'] + 1)

        def shutdown():
            state.update(on=False, shutdowns=state['shutdowns'] + 1)

        monkeypatch.setattr(ray, 'init', init)
        monkeypatch.setattr(ray, 'shutdown', shutdown, raising=False)
        monkeypatch.setattr(ray, 'is_initialized', lambda: state['on'])
        return state

    def _revs(self, repos):
        return [f"{repos.dbx_shas[i]}:{repos.proj_shas[i]}" for i in range(2)]

    def test_remotes_at_two_revisions_share_one_connection(self, ray_, repos, stub_ray):
        envs = []
        for rev in self._revs(repos):
            databits_mod.remote(revision=rev)
            envs.append(stub_ray['actor_env']['env_vars']['DBX_PIN_REVISION'])
        assert (ray_['inits'], ray_['shutdowns']) == (1, 0)
        assert envs == self._revs(repos), "each actor pinned to its own revision"

    def test_dbx_ray_keeps_the_connection_too(self, ray_, repos, monkeypatch):
        monkeypatch.setenv('DBX_RAY', 'auto')
        for rev in self._revs(repos):
            databits_mod.remote(revision=rev)
        assert (ray_['inits'], ray_['shutdowns']) == (1, 0)

    def test_no_address_rides_the_one_made_for_an_address(self, ray_, repos):
        databits_mod.remote(address='auto')
        databits_mod.remote(revision=self._revs(repos)[0])
        assert (ray_['inits'], ray_['shutdowns']) == (1, 0)

    def test_another_cluster_is_another_connection(self, ray_):
        databits_mod.remote(address='auto')
        databits_mod.remote(address='ray://elsewhere:10001')
        assert (ray_['inits'], ray_['shutdowns']) == (2, 1)

    def test_a_connection_made_elsewhere_is_kept_unpinned(self, ray_):
        ray_['on'] = True                     # e.g. dbx.gpu.run's, or the user's own ray.init
        databits_mod.remote()
        assert (ray_['inits'], ray_['shutdowns']) == (0, 0)

    def test_a_connection_made_elsewhere_has_no_shim_and_is_refused(self, ray_, repos):
        """Whatever made it may still be using it: dbx does not shut it down, it says so."""
        ray_['on'] = True
        with pytest.raises(databits_mod.RayReconnectError, match='not by dbx'):
            databits_mod.remote(revision=self._revs(repos)[0])
        assert (ray_['inits'], ray_['shutdowns']) == (0, 0)

    def test_another_cluster_is_refused_while_a_remote_lives(self, ray_):
        held = databits_mod.remote(address='auto')
        with pytest.raises(databits_mod.RayReconnectError, match='1 Remote'):
            databits_mod.remote(address='ray://elsewhere:10001')
        assert (ray_['inits'], ray_['shutdowns']) == (1, 0)
        del held
        databits_mod.remote(address='ray://elsewhere:10001')
        assert (ray_['inits'], ray_['shutdowns']) == (2, 1), "once no Remote lives on it"

    def test_with_ray_names_the_cluster(self, ray_, stub_ray):
        with databits_mod.Ray('ray://scoped:10001'):
            assert databits_mod.Ray.current().address == 'ray://scoped:10001'
            databits_mod.remote()
        assert stub_ray['init']['address'] == 'ray://scoped:10001'
        assert databits_mod.Ray.current() is databits_mod.DEFAULT_RAY

    def test_default_ray_reads_dbx_ray_when_it_connects(self, monkeypatch):
        monkeypatch.setenv('DBX_RAY', 'ray://from-env:10001')
        assert databits_mod.DEFAULT_RAY.address == 'ray://from-env:10001'


# ---------------------------------------------------------------------------
# A worker checks its pin, and says so loudly
# ---------------------------------------------------------------------------

class TestCheckPinned:
    """Pinning happens before dbx imports, unchecked -- a setup hook Ray ignores leaves a worker on another dbx that
    still claims the revision. `check_pinned` is the worker's check that its code is the revision's."""

    def _module(self, path):
        return types.SimpleNamespace(__file__=os.path.join(path, 'dbx', '__init__.py'))

    def test_unpinned_is_not_checked(self, monkeypatch):
        monkeypatch.delenv('DBX_PINNED_REVISION', raising=False)
        databits_mod.check_pinned(self._module('/nowhere'))

    def test_dbx_from_a_checkout_at_the_revision_passes(self, repos, monkeypatch):
        rev = f"{repos.dbx_shas[0]}:{repos.proj_shas[0]}"
        dbx_pin, proj_pin = databits_mod.gitpinrepos(rev, gitrepo=repos.gitrepo, pin_root=repos.pin_root)
        monkeypatch.setenv('DBX_PINNED_REVISION', rev)
        monkeypatch.setattr(sys, 'path', [proj_pin, *sys.path])
        databits_mod.check_pinned(self._module(dbx_pin))

    def test_dbx_from_elsewhere_fails(self, repos, monkeypatch):
        rev = f"{repos.dbx_shas[0]}:{repos.proj_shas[0]}"
        monkeypatch.setenv('DBX_PINNED_REVISION', rev)
        with pytest.raises(databits_mod.PinMismatchError, match='dbx is imported from'):
            databits_mod.check_pinned(self._module('/site-packages'))

    def test_dbx_from_a_checkout_at_another_revision_fails(self, repos, monkeypatch):
        dbx_pin, _ = databits_mod.gitpinrepos(f"{repos.dbx_shas[1]}:{repos.proj_shas[0]}",
                                             gitrepo=repos.gitrepo, pin_root=repos.pin_root)
        monkeypatch.setenv('DBX_PINNED_REVISION', f"{repos.dbx_shas[0]}:{repos.proj_shas[0]}")
        with pytest.raises(databits_mod.PinMismatchError):
            databits_mod.check_pinned(self._module(dbx_pin))

    def test_no_project_checkout_on_the_path_fails(self, repos, monkeypatch):
        rev = f"{repos.dbx_shas[0]}:{repos.proj_shas[0]}"
        dbx_pin, _ = databits_mod.gitpinrepos(rev, gitrepo=repos.gitrepo, pin_root=repos.pin_root)
        monkeypatch.setenv('DBX_PINNED_REVISION', rev)
        monkeypatch.setattr(sys, 'path', [p for p in sys.path if 'pins' not in p])
        with pytest.raises(databits_mod.PinMismatchError, match='no checkout at'):
            databits_mod.check_pinned(self._module(dbx_pin))

    def test_remote_raises_when_the_worker_is_pinned_otherwise(self, repos, stub_ray, monkeypatch):
        monkeypatch.setattr(databits_mod, 'DBX_USE_WORK_REPO', repos.gitrepo)
        monkeypatch.setattr(stub_ray['ray'], 'get', lambda ref: 'something-else')
        with pytest.raises(databits_mod.PinMismatchError, match='got one pinned to something-else'):
            databits_mod.remote(revision=f"{repos.dbx_shas[0]}:{repos.proj_shas[0]}", pin_root=repos.pin_root)
