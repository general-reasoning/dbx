"""dbx.anyscale without a workspace: which one the arguments and environment name, and the url and config made of it."""
import asyncssh.config
import pytest

from dbx import anyscale


@pytest.fixture
def no_anyscale_env(monkeypatch):
    for var in (*anyscale.ID_ENV_VARS, anyscale.NAME_ENV_VAR, anyscale.CLOUD_ENV_VAR, anyscale.PROJECT_ENV_VAR):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_given_arguments_win_over_the_environment(no_anyscale_env):
    no_anyscale_env.setenv('ANYSCALE_WORKSPACE_ID', 'expwrk_env')
    assert anyscale._workspace_args_('expwrk_arg', None, None, None)['id'] == 'expwrk_arg'
    assert anyscale._workspace_args_(None, 'ws', None, None) == dict(id=None, name='ws', cloud=None, project=None)


def test_the_environment_names_the_workspace(no_anyscale_env):
    no_anyscale_env.setenv('ANYSCALE_EXPERIMENTAL_WORKSPACE_ID', 'expwrk_inside')
    assert anyscale._workspace_args_(None, None, None, None)['id'] == 'expwrk_inside'
    no_anyscale_env.setenv('ANYSCALE_WORKSPACE_ID', 'expwrk_outside')
    assert anyscale._workspace_args_(None, None, None, None)['id'] == 'expwrk_outside'


def test_cloud_and_project_come_only_with_a_name(no_anyscale_env):
    no_anyscale_env.setenv('ANYSCALE_CLOUD', 'c')
    no_anyscale_env.setenv('ANYSCALE_PROJECT', 'p')
    assert anyscale._workspace_args_('expwrk_x', None, None, None) == dict(id='expwrk_x', name=None, cloud=None, project=None)
    no_anyscale_env.setenv('ANYSCALE_WORKSPACE_NAME', 'ws')
    assert anyscale._workspace_args_(None, None, None, 'q') == dict(id=None, name='ws', cloud='c', project='q')


def test_no_workspace_is_an_error(no_anyscale_env):
    with pytest.raises(ValueError, match='No Anyscale workspace'):
        anyscale._workspace_args_(None, None, None, None)


def test_datalake_is_an_ssh_url_on_the_lowercased_id(no_anyscale_env):
    assert anyscale.datalake('/mnt/lake/', id='expwrk_AB1') == 'ssh://expwrk_ab1/mnt/lake'
    with pytest.raises(ValueError, match='absolute path'):
        anyscale.datalake('lake', id='expwrk_ab1')


def test_ssh_config_reads_as_asyncssh_reads_it(tmp_path):
    path = tmp_path / 'config'
    path.write_text(anyscale.SSH_CONFIG.format(host='expwrk_ab1', key='/k.pem', known_hosts='/kh',
                                               proxy='python -m dbx.anyscale proxy expwrk_ab1'))
    config = asyncssh.config.SSHClientConfig.load(None, [str(path)], False, False, False, 'me', (), 'expwrk_ab1', 22)
    assert config.get('User') == 'ray'
    assert config.get('IdentityFile') == ['/k.pem']
    assert config.get('UserKnownHostsFile') == ['/kh']
    assert config.get('ProxyCommand') == 'python -m dbx.anyscale proxy expwrk_ab1'
