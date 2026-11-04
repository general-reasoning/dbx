"""dbx.gpu: a function farmed out as one Ray task per call, each logged to its own file.

Ray itself is faked: what is checked is what `run` asks of it, and what a task does with its call.
"""
import sys
import types

import pytest

from dbx import anyscale, gpu


class FakeRay(types.SimpleNamespace):
    """Records `init` and runs each `remote` task in-process."""

    def __init__(self):
        super().__init__(inits=[], resources=[])

    def init(self, **kwargs):
        self.inits.append(kwargs)

    def remote(self, **resources):
        self.resources.append(resources)

        def wrap(fn):
            return types.SimpleNamespace(remote=lambda *a: fn(*a))
        return wrap

    def get(self, results):
        return list(results)


@pytest.fixture
def ray(monkeypatch):
    fake = FakeRay()
    monkeypatch.setitem(sys.modules, 'ray', fake)
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
    assert ray.inits[0]['runtime_env'] == {'py_executable': sys.executable, 'env_vars': {'X': '1'}}
    assert (tmp_path / 'task1.log').read_text() == 'shard 1\nprinted\n'


def test_on_anyscale_the_working_dir_is_given(ray, tmp_path, monkeypatch):
    monkeypatch.setattr(anyscale, 'present', lambda: True)
    monkeypatch.setenv('ANYSCALE_WORKING_DIR', '/tmp/wd')
    gpu.run(_shard_, [dict(shard=0)], log_dir=str(tmp_path))
    assert ray.inits[0]['runtime_env']['working_dir'] == '/tmp/wd'


def test_a_tasks_exception_is_raised_here(ray, tmp_path, monkeypatch):
    monkeypatch.setattr(anyscale, 'present', lambda: False)
    with pytest.raises(KeyError):
        gpu.run(_shard_, [dict(shard=-1)], log_dir=str(tmp_path))
