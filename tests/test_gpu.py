"""dbx.gpu: a function farmed out as one Ray task per call, each logged to its own file.

Ray itself is faked: what is checked is what `run` asks of it, and what a task does with its call.
"""
import sys
import types

import pytest

import weakref

from dbx import anyscale, gpu
from dbx.databits import Ray


class FakeRay(types.SimpleNamespace):
    """Records `init` and each task's resources and runtime_env, and runs each `remote` task in-process."""

    def __init__(self):
        super().__init__(inits=[], resources=[], task_envs=[])

    def init(self, **kwargs):
        self.inits.append(kwargs)

    def is_initialized(self):
        return bool(self.inits)

    def remote(self, **resources):
        self.resources.append(resources)

        def wrap(fn):
            task = types.SimpleNamespace(remote=lambda *a: fn(*a))
            task.options = lambda runtime_env=None: self.task_envs.append(runtime_env) or task
            return task
        return wrap

    def get(self, results):
        return list(results)


@pytest.fixture
def ray(monkeypatch):
    fake = FakeRay()
    monkeypatch.setitem(sys.modules, 'ray', fake)
    monkeypatch.setattr(Ray, '_connection_', None)
    monkeypatch.setattr(Ray, '_remotes_', weakref.WeakSet())
    monkeypatch.delenv('DBX_RAY', raising=False)
    return fake


def _shard_(shard, log):
    log(f"shard {shard}")
    print('printed')
    if shard < 0:
        raise KeyError(shard)
    return shard * 10


def test_each_call_is_a_task_with_its_resources_and_its_own_log(ray, tmp_path, monkeypatch):
    monkeypatch.setattr(anyscale, 'present', lambda: False)
    got = gpu.run(_shard_, [dict(shard=0), dict(shard=1)], log_dir=str(tmp_path), n_gpus=1, n_cpus=6,
                  env={'X': '1'})
    assert got == [0, 10]
    assert ray.resources == [dict(num_gpus=1, num_cpus=6)]
    assert ray.task_envs == [{'py_executable': sys.executable, 'env_vars': {'X': '1'}}], "each task's own"
    assert ray.inits[0]['address'] == 'auto' and 'working_dir' not in ray.inits[0]['runtime_env']
    assert (tmp_path / 'task1.log').read_text() == 'shard 1\nprinted\n'


def test_on_anyscale_the_working_dir_is_given(ray, tmp_path, monkeypatch):
    monkeypatch.setattr(anyscale, 'present', lambda: True)
    monkeypatch.setenv('ANYSCALE_WORKING_DIR', '/tmp/wd')
    gpu.run(_shard_, [dict(shard=0)], log_dir=str(tmp_path))
    assert ray.inits[0]['runtime_env']['working_dir'] == '/tmp/wd'


def test_a_second_run_rides_the_connection(ray, tmp_path, monkeypatch):
    monkeypatch.setattr(anyscale, 'present', lambda: False)
    gpu.run(_shard_, [dict(shard=0)], log_dir=str(tmp_path), env={'A': '1'})
    gpu.run(_shard_, [dict(shard=0)], log_dir=str(tmp_path), env={'B': '2'})
    assert len(ray.inits) == 1, "one connection; each run's env is its tasks'"
    assert [e['env_vars'] for e in ray.task_envs] == [{'A': '1'}, {'B': '2'}]


def test_a_tasks_exception_is_raised_here(ray, tmp_path, monkeypatch):
    monkeypatch.setattr(anyscale, 'present', lambda: False)
    with pytest.raises(KeyError):
        gpu.run(_shard_, [dict(shard=-1)], log_dir=str(tmp_path))
