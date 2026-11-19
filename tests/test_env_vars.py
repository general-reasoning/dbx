"""`dbx.env`: the environment variables dbx reads, each read when it is accessed."""
import os
import subprocess
import sys

import pytest

import dbx
import dbx.env as envvars


def test_a_variable_is_read_when_accessed(monkeypatch):
    monkeypatch.setenv('DBX_RAY', 'auto')
    assert envvars.DBX_RAY == 'auto'
    monkeypatch.setenv('DBX_RAY', 'ray://head:10001')
    assert envvars.DBX_RAY == 'ray://head:10001'


def test_unset_or_empty_is_none(monkeypatch):
    monkeypatch.delenv('DBX_RAY', raising=False)
    assert envvars.DBX_RAY is None
    monkeypatch.setenv('DBX_RAY', '')
    assert envvars.DBX_RAY is None


def test_a_name_not_listed_is_an_attribute_error():
    with pytest.raises(AttributeError, match='not a dbx environment variable'):
        envvars.DBX_NO_SUCH_THING


def test_a_family_member_is_read_as_a_listed_one(monkeypatch):
    monkeypatch.setenv('DBX_LOG_DEBUG', 'True')
    assert envvars.DBX_LOG_DEBUG == 'True'


def test_every_variable_is_listed_and_documented():
    assert all(name.startswith('DBX_') and doc for name, doc in envvars.VARS.items())
    assert set(envvars.VARS) <= set(dir(envvars))


def test_dbx_env_is_the_module_and_still_makes_speclines():
    """`dbx.env` was the specline function before it was a module: called, it still is."""
    from dbx import env
    assert env is envvars and dbx.env is envvars
    assert dbx.env('DATA_ROOT') == "$dbx.getenv('DATA_ROOT')" == dbx.databits.env('DATA_ROOT')
    assert dbx.env('X', 'd') == "$dbx.getenv('X', 'd')"


def test_the_datalake_follows_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv('DBX_DATALAKE', str(tmp_path))
    assert dbx.default_datalake() == str(tmp_path)


def test_dbx_dirty_sets_its_pair_at_import(tmp_path):
    env = {**os.environ, 'DBX_DIRTY': 'True', 'DBX_USE_WORK_REPO': 'True'}
    env.pop('DBX_DIRTY_REPO_OK', None)
    out = subprocess.run([sys.executable, '-c', 'import dbx.env as e; print(e.DBX_USE_WORK_REPO, e.DBX_DIRTY_REPO_OK)'],
                         env=env, cwd=tmp_path, capture_output=True, text=True, check=True).stdout.split()
    assert out == ['False', 'True']
