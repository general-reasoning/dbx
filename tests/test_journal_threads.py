"""Journal readers share one filesystem between their threads -- except where its one connection cannot be shared."""
import fsspec
import fsspec.asyn

from dbx.journals import _threads_


class _SFTPLike_:
    protocol = ('sftp', 'ssh')


def test_a_local_filesystem_is_read_by_as_many_threads_as_asked():
    assert _threads_(fsspec.filesystem('file'), 8) == 8


def test_an_sftp_filesystem_is_read_serially():
    # paramiko's one SFTP client deadlocks when several threads send it requests at once.
    assert _threads_(_SFTPLike_(), 8) == 1


def test_an_async_ssh_filesystem_is_read_by_as_many_threads_as_asked():
    # sshfs runs every request on its own event loop, over a pool of SFTP channels: threads share it.
    class _SSHFSLike_(fsspec.asyn.AsyncFileSystem):
        protocol = ('ssh', 'sftp')
    assert _threads_(_SSHFSLike_(), 8) == 8
