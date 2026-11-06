"""dbx.kube: the Job a command or a pipeline's call becomes, its submission journaled, and its jobs and queues.

kubectl is never run: `dbx.kube._kubectl_` is replaced by a fake that records what it is handed and
answers as the cluster would.
"""
import ast
import json
import os
import subprocess
import sys
from dataclasses import dataclass

import pytest

import dbx
from dbx import cluster as dbxcluster, kube
from dbx.datablocks import DATAFILE, Datablock
from dbx.journals import PipeJournal
from dbx.pipelines import Pipeline


@pytest.fixture
def cluster():
    return kube.KubeCluster('testc', context='ctx', namespace='ns', queue='gpu-q', cpu_queue='cpu-q',
                        priority='normal', mounts={'/mnt/team': 'team-pvc', '/mnt/data': 'data-pvc'},
                        readonly=('/mnt/data',), n_cpus_per_gpu=10, mem_gib_per_gpu=100, image='img:1', user='me')


class _EP_:
    def __init__(self, obj, name='ep'):
        self.obj, self.name = obj, name

    def load(self):
        return self.obj


@pytest.fixture
def entry_points(monkeypatch):
    """The installed `dbx.clusters` entry points, as a list to put objects in."""
    eps = []
    monkeypatch.setattr(dbxcluster.metadata, 'entry_points',
                        lambda group: [_EP_(o) for o in eps] if group == 'dbx.clusters' else [])
    return eps


class FakeKubectl:
    """What kubectl was asked, and the Job status it reports."""

    def __init__(self, conditions=(), fail_create=False):
        self.calls, self.created, self.conditions, self.fail_create = [], [], list(conditions), fail_create
        self.deleted = set()

    def __call__(self, context, *args, input=None):
        self.calls.append((context, args))
        if 'delete' in args:
            self.deleted.update(a.removeprefix('job/') for a in args if a.startswith('job/'))
            return ''
        if 'get' in args and any(a.removeprefix('job/') in self.deleted for a in args if a.startswith('job/')):
            raise RuntimeError('Error from server (NotFound): jobs.batch not found')
        if 'create' in args:
            if self.fail_create:
                raise RuntimeError("kubectl create: forbidden")
            self.created.append(json.loads(input))
            return f"job.batch/{self.created[-1]['metadata']['name']}\n"
        if 'get' in args and 'workload' in args:
            return json.dumps({'items': [{'metadata': {'ownerReferences': [{'kind': 'Job', 'name': j['metadata']['name']}]},
                                          'status': {'conditions': [{'type': 'QuotaReserved', 'status': 'False',
                                                                     'message': 'insufficient quota for cpu'}]}}
                                         for j in self.created if self.conditions == ['Pending']]})
        if 'get' in args and 'job' in args:
            return json.dumps({'items': [{**j, 'status': {}} for j in self.created]})
        if 'get' in args:
            if self.conditions == ['Pending']:
                return json.dumps({'spec': {'suspend': True}, 'status': {}})
            return json.dumps({'spec': {'suspend': False}, 'status': {
                'active': 1, 'conditions': [{'type': t, 'status': 'True'} for t in self.conditions]}})
        if 'logs' in args:
            return "line 1\nTraceback: boom\n"
        return ''


@pytest.fixture
def kubectl(monkeypatch):
    fake = FakeKubectl()
    monkeypatch.setattr(kube, '_kubectl_', fake)
    return fake


def _container_(manifest):
    return manifest['spec']['template']['spec']['containers'][0]


def _env_(manifest):
    return {e['name']: e['value'] for e in _container_(manifest)['env']}


# --------------------------------------------------------------------------------------------- #
# The Job
# --------------------------------------------------------------------------------------------- #

class TestTheJob:
    def test_a_gpu_job_is_sized_by_the_clusters_share_and_queued_suspended(self, cluster):
        m = cluster.job('echo hi', n_gpus=2, name='My Thing')
        res = _container_(m)['resources']['requests']
        assert (res['cpu'], res['memory'], res['nvidia.com/gpu']) == ('20', '200Gi', '2')
        assert m['metadata']['labels']['kueue.x-k8s.io/queue-name'] == 'gpu-q'
        assert m['metadata']['labels']['kueue.x-k8s.io/priority-class'] == 'normal'
        assert m['metadata']['name'].startswith('me-my-thing-') and m['metadata']['namespace'] == 'ns'
        assert m['spec']['suspend'] is True and m['spec']['backoffLimit'] == 0
        assert _env_(m)['DBX_KUBE_COMMAND'] == 'echo hi' and m['metadata']['annotations']['dbx/command'] == 'echo hi'

    def test_a_job_without_gpus_goes_to_the_cpu_queue(self, cluster):
        m = cluster.job('echo hi')
        assert m['metadata']['labels']['kueue.x-k8s.io/queue-name'] == 'cpu-q'
        assert _container_(m)['resources']['requests']['nvidia.com/gpu'] == '0'

    def test_the_claims_mount_where_the_cluster_says_read_only_where_it_says(self, cluster):
        m = cluster.job('true')
        mounts = {v['mountPath']: v for v in _container_(m)['volumeMounts']}
        claims = {v['name']: v for v in m['spec']['template']['spec']['volumes']}
        assert claims[mounts['/mnt/team']['name']]['persistentVolumeClaim']['claimName'] == 'team-pvc'
        assert mounts['/mnt/data'].get('readOnly') and not mounts['/mnt/team'].get('readOnly')

    def test_off_anyscale_nothing_of_anyscales_is_assumed(self, cluster):
        m = cluster.job('true', ray=True)
        script = _container_(m)['command'][2]
        assert 'securityContext' not in m['spec']['template']['spec']
        assert not any(k.startswith('ANYSCALE') for k in _env_(m))
        assert '\nray start --head' in script and '/home/ray' not in script

    def test_anyscale_adds_its_user_its_ray_and_its_working_dir(self, cluster):
        m = cluster.job('true', anyscale=True, ray=True)
        script = _container_(m)['command'][2]
        assert m['spec']['template']['spec']['securityContext'] == {'runAsUser': 1000, 'runAsGroup': 100}
        assert _env_(m)['ANYSCALE_WORKING_DIR'] == dbx.anyscale.WORKING_DIR
        assert '/home/ray/venv/bin/ray start --head' in script and 'mkdir -p "$ANYSCALE_WORKING_DIR"' in script

    def test_no_image_is_an_error_that_says_where_one_comes_from(self, cluster):
        cluster.image = None
        with pytest.raises(ValueError, match='image='):
            cluster.job('true')

    def test_the_venv_is_activated_and_the_command_runs_last(self, cluster):
        script = _container_(cluster.job('true', venv='/mnt/team/venv', setup=['echo setup']))['command'][2]
        lines = script.splitlines()
        assert lines.index('echo setup') < lines.index('source /mnt/team/venv/bin/activate')
        assert lines[-1] == 'exec bash -c "$DBX_KUBE_COMMAND"'

    def test_the_clusters_options_are_defaults_and_its_env_is_added_to(self, cluster):
        cluster.options = {'venv': '/v', 'env': {'A': '1'}, 'n_gpus': 1}
        m = cluster.job('true', env={'B': '2'}, n_gpus=4)
        assert _env_(m)['A'] == '1' and _env_(m)['B'] == '2'
        assert 'source /v/bin/activate' in _container_(m)['command'][2]
        assert _container_(m)['resources']['requests']['nvidia.com/gpu'] == '4'


class TestRevision:
    @pytest.fixture
    def repos(self, tmp_path, monkeypatch):
        paths = []
        for name in ('dbx', 'proj'):
            d = tmp_path / name
            d.mkdir()
            for cmd in (['init', '-q'], ['-c', 'user.email=t@t', '-c', 'user.name=t', 'commit', '-q',
                                         '--allow-empty', '-m', 'c']):
                subprocess.run(['git', '-C', str(d), *cmd], check=True)
            paths.append(str(d))
        monkeypatch.setattr(dbx.databits, 'DBX_GIT_REPO', ':'.join(paths))
        return paths

    def _sha_(self, repo):
        return subprocess.run(['git', '-C', repo, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()

    def test_both_repos_are_cloned_at_their_commits_dbx_first_on_the_path(self, cluster, repos):
        m = cluster.job('true', revision='HEAD:HEAD')
        script = _container_(m)['command'][2]
        dsha, psha = self._sha_(repos[0]), self._sha_(repos[1])
        assert f"checkout --quiet {dsha}" in script and f"checkout --quiet {psha}" in script
        exports = [l for l in script.splitlines() if l.startswith('export PYTHONPATH=')]
        assert 'dbx' in exports[-1] and 'proj' in exports[0]
        assert m['metadata']['annotations']['dbx/revision'] == f"{repos[0]}@{dsha} {repos[1]}@{psha}"

    def test_revision_true_refuses_a_dirty_checkout(self, cluster, repos):
        with open(os.path.join(repos[1], 'f'), 'w') as f:
            f.write('x')
        subprocess.run(['git', '-C', repos[1], 'add', 'f'], check=True)
        with pytest.raises(ValueError, match='commit first'):
            cluster.job('true', revision=True)


# --------------------------------------------------------------------------------------------- #
# Workdirs
# --------------------------------------------------------------------------------------------- #

class TestWorkdirs:
    def test_a_path_maps_to_its_url_under_the_clusters_storage(self, cluster):
        cluster.storage = {'/mnt/team': 'sftp://host/data'}
        assert cluster.url('/mnt/team/a/b') == 'sftp://host/data/a/b'
        with pytest.raises(ValueError):
            cluster.url('/elsewhere')
        cluster.storage = {}
        assert cluster.url('/mnt/team/a') == '/mnt/team/a'

    def test_storage_options_are_a_dict_or_a_function_named_as_data(self, cluster):
        cluster.storage_options = {'username': 'u'}
        assert cluster.fsspec_options() == {'username': 'u'}
        cluster.storage_options = 'tests.test_kube:_sftp_options_'
        assert cluster.fsspec_options() == {'username': 'from-a-function'}
        assert dbxcluster._rebuild_({'target_spec': dbxcluster._spec_(cluster)}) == cluster
        old = dbxcluster._spec_(cluster) | {'class': 'dbx.kube:Cluster'}    # as journaled before the rename
        assert dbxcluster._rebuild_({'target_spec': old}) == cluster

    def test_fsspec_defaults_hold_only_while_inside(self, cluster):
        import fsspec
        cluster.storage, cluster.storage_options = {'/mnt/team': 'sftp://host/data'}, {'username': 'u'}
        before = fsspec.config.conf.get('sftp')
        with cluster.fsspec_defaults():
            assert fsspec.config.conf['sftp'] == {'username': 'u'}
        assert fsspec.config.conf.get('sftp') == before

    def test_a_synced_workdir_is_opened_with_the_clusters_options(self, cluster, kubectl, monkeypatch, tmp_path):
        seen = []
        monkeypatch.setattr(kube, '_sync_', lambda src, url, *, storage_options, progress=None: seen.append((url, storage_options)) or
                            dict(files=0, bytes=0, skipped=0))
        cluster.storage, cluster.storage_options = {'/mnt/team': 'sftp://host/data'}, {'username': 'u'}
        cluster.submit('true', workdirs={str(tmp_path): '/mnt/team/code'})
        assert seen == [('sftp://host/data/code', {'username': 'u'})]

    def test_sync_copies_what_changed_keeps_modes_and_links(self, tmp_path):
        src, dst = tmp_path / 'src', tmp_path / 'dst'
        (src / 'bin').mkdir(parents=True)
        (src / 'bin' / 'tool').write_text('#!/bin/sh\n')
        os.chmod(src / 'bin' / 'tool', 0o755)
        (src / 'data.txt').write_text('abc')
        os.symlink('data.txt', src / 'link')
        first = kube._sync_(str(src), f"file://{dst}")
        assert first['files'] == 2 and os.stat(dst / 'bin' / 'tool').st_mode & 0o777 == 0o755
        assert os.readlink(dst / 'link') == 'data.txt'
        assert kube._sync_(str(src), f"file://{dst}") == dict(files=0, bytes=0, skipped=2)
        (src / 'data.txt').write_text('abcd')
        assert kube._sync_(str(src), f"file://{dst}")['files'] == 1 and (dst / 'data.txt').read_text() == 'abcd'

    def test_a_clusters_sync_goes_to_its_storage_url_or_nowhere_when_it_mounts_ours(self, tmp_path):
        (tmp_path / 'src').mkdir()
        (tmp_path / 'src' / 'a.txt').write_text('a')
        stored = kube.KubeCluster('s', image='i', storage={str(tmp_path / 'src'): f"file://{tmp_path / 'there'}"})
        assert stored.sync(str(tmp_path / 'src'))['files'] == 1 and (tmp_path / 'there' / 'a.txt').read_text() == 'a'
        assert kube.KubeCluster('m', image='i').sync(str(tmp_path / 'src')) == dict(files=0, bytes=0, skipped=None)

    def test_a_cluster_that_mounts_our_paths_takes_a_workdir_only_where_it_is(self, cluster, kubectl, tmp_path):
        cluster.submit('true', workdirs={str(tmp_path): str(tmp_path)})
        with pytest.raises(ValueError, match='there already'):
            cluster.submit('true', workdirs={str(tmp_path): '/mnt/team/x'})

    def test_a_cluster_with_its_own_storage_is_synced_before_the_job(self, cluster, kubectl, tmp_path):
        (tmp_path / 'src').mkdir()
        (tmp_path / 'src' / 'f').write_text('x')
        cluster.storage = {'/mnt/team': f"file://{tmp_path / 'remote'}"}
        run = cluster.submit('true', workdirs={str(tmp_path / 'src'): '/mnt/team/code'})
        assert (tmp_path / 'remote' / 'code' / 'f').read_text() == 'x'
        assert run.record['workdirs'][str(tmp_path / 'src')]['files'] == 1


class TestPackedWorkdirs:
    """pack=True: each workdir sent as one archive, named by its tree's digest, which the Job unpacks first."""

    @pytest.fixture
    def tree(self, tmp_path):
        src = tmp_path / 'src'
        (src / 'bin').mkdir(parents=True)
        (src / 'bin' / 'tool').write_text('#!/bin/sh\n')
        os.chmod(src / 'bin' / 'tool', 0o755)
        (src / 'data.txt').write_text('abc')
        os.symlink('data.txt', src / 'link')
        (src / '__pycache__').mkdir()
        (src / '__pycache__' / 'x.pyc').write_text('bytecode')
        return src

    @pytest.fixture
    def remote(self, cluster, tmp_path):
        cluster.storage = {'/mnt/team': f"file://{tmp_path / 'remote'}"}
        return tmp_path / 'remote'

    def test_the_tree_goes_as_one_archive_the_job_unpacks_first(self, cluster, kubectl, tree, remote):
        run = cluster.submit('true', workdirs={str(tree): '/mnt/team/code'}, pack=True)
        [(src, done)] = run.record['workdirs'].items()
        assert src == str(tree) and done['uploaded'] and done['archive'].startswith('/mnt/team/.dbx-packs/src-')
        assert (remote / '.dbx-packs' / os.path.basename(done['archive'])).exists()
        assert not (remote / 'code').exists()          # nothing copied file by file
        script = _container_(kubectl.created[0])['command'][2]
        assert script.index(f"dbx_unpack {done['archive']} /mnt/team/code {done['digest']}") < script.index('exec bash')

    def test_an_unchanged_tree_is_not_sent_again_and_a_change_is(self, cluster, kubectl, tree, remote):
        first = cluster.submit('true', workdirs={str(tree): '/mnt/team/code'}, pack=True)
        again = cluster.submit('true', workdirs={str(tree): '/mnt/team/code'}, pack=True)
        (tree / '__pycache__' / 'y.pyc').write_text('more bytecode')          # left out: no new archive
        cached = cluster.submit('true', workdirs={str(tree): '/mnt/team/code'}, pack=True)
        (tree / 'data.txt').write_text('abcd')
        changed = cluster.submit('true', workdirs={str(tree): '/mnt/team/code'}, pack=True)
        d = [r.record['workdirs'][str(tree)] for r in (first, again, cached, changed)]
        assert [x['uploaded'] for x in d] == [True, False, False, True]
        assert d[0]['digest'] == d[1]['digest'] == d[2]['digest'] != d[3]['digest']

    def test_the_unpack_makes_the_target_the_tree_once(self, cluster, kubectl, tree, remote, tmp_path):
        run = cluster.submit('true', workdirs={str(tree): '/mnt/team/code'}, pack=True)
        p = run.record['workdirs'][str(tree)]
        target = tmp_path / 'there' / 'code'
        target.mkdir(parents=True)
        (target / 'stale.txt').write_text('from before')
        lines = kube._unpack_lines_([dict(archive=str(remote / '.dbx-packs' / os.path.basename(p['archive'])),
                                          target=str(target), digest=p['digest'])])
        script = '\n'.join(['set -euo pipefail', *lines])
        first = subprocess.run(['bash', '-c', script], capture_output=True, text=True, check=True)
        assert 'unpacked' in first.stdout and (target / 'data.txt').read_text() == 'abc'
        assert os.stat(target / 'bin' / 'tool').st_mode & 0o777 == 0o755 and os.readlink(target / 'link') == 'data.txt'
        assert not (target / 'stale.txt').exists() and not (target / '__pycache__').exists()
        assert (target / '.dbx-pack').read_text().strip() == p['digest']
        second = subprocess.run(['bash', '-c', script], capture_output=True, text=True, check=True)
        assert 'unpacked' not in second.stdout

    def test_a_cluster_may_pack_by_default_and_a_call_may_not(self, cluster, kubectl, tree, remote):
        cluster.options = {'pack': True}
        packed = cluster.submit('true', workdirs={str(tree): '/mnt/team/code'})
        synced = cluster.submit('true', workdirs={str(tree): '/mnt/team/code'}, pack=False)
        assert packed.record['workdirs'][str(tree)]['uploaded'] and 'files' in synced.record['workdirs'][str(tree)]

    def test_a_target_holding_a_mount_or_the_datalake_is_refused(self, cluster, kubectl, tree, remote, monkeypatch):
        with pytest.raises(ValueError, match='would be replaced'):
            cluster.submit('true', workdirs={str(tree): '/mnt/team'}, pack=True)
        monkeypatch.setenv('DBX_DATALAKE', '/mnt/team/me/datalake')
        with pytest.raises(ValueError, match='would be replaced'):
            cluster.submit('true', workdirs={str(tree): '/mnt/team/me'}, pack=True)

    def test_a_cluster_mounting_our_paths_is_not_packed(self, cluster, kubectl, tmp_path):
        run = cluster.submit('true', workdirs={str(tmp_path): str(tmp_path)}, pack=True)
        assert 'dbx_unpack' not in _container_(kubectl.created[0])['command'][2]
        assert run.record['workdirs'][str(tmp_path)]['skipped'] is None


# --------------------------------------------------------------------------------------------- #
# Submitting
# --------------------------------------------------------------------------------------------- #

class Leaf(Datablock):
    TOPICS = {'x': DATAFILE('x.txt')}

    @dataclass
    class VAR(Datablock.VAR):
        n: int = 1

    def __build__(self):
        with open(self.path('x', ensure_dirpath=True), 'w') as f:
            f.write(str(self.var.n))


class Single(Pipeline):
    @dataclass
    class VAR(Pipeline.VAR):
        n: int = 1

    def __create__(self):
        return Leaf(spec={'n': self.var.n}, datalake=self.datalake, **self.operational)


def _printed_(fn) -> str:
    import contextlib
    import io
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        fn()
    return out.getvalue()


def _pipe_events_(pipeline):
    """*pipeline*'s own `submit:*` events in the `PipeJournal`, oldest first."""
    try:
        j = PipeJournal.read(f"{PipeJournal.ANCHOR}/{pipeline.anchor}", datalake=pipeline.datalake, index=None,
                             hash=pipeline.hash, event='^job:')
    except FileNotFoundError:
        return []
    return [] if j is None or len(j) == 0 else list(j.sort_values('datetime')['event'])


def _events_(dblk):
    """*dblk*'s own journal events, oldest first -- its anchor's journal is its whole class's."""
    try:
        j = dblk.buildjournal(hash=dblk.hash)
    except FileNotFoundError:  # No journal at all: nothing was ever written for it.
        return []
    return [] if j is None or len(j) == 0 else list(j.sort_values('datetime')['event'])


class TestSubmit:
    def test_a_dry_run_contacts_nothing_and_journals_nothing(self, cluster, kubectl, tmp_path):
        b = Leaf(datalake=str(tmp_path))
        run = cluster.submit('true', block=b, dry_run=True)
        assert kubectl.calls == [] and run.job is None and run.status() == 'dry-run'
        assert _events_(b) == [] and run.record['manifest']['kind'] == 'Job'

    def test_a_submission_is_journaled_with_its_whole_record(self, cluster, kubectl, tmp_path):
        b = Leaf(datalake=str(tmp_path))
        run = cluster.submit('echo hi', block=b, n_gpus=1)
        assert kubectl.created[0]['metadata']['name'] == run.job
        j = b.buildjournal()
        [row] = j[j['event'] == 'job:submit'].to_dict('records')
        with open(row['note']) as f:
            record = json.load(f)
        assert record['job'] == run.job and record['cluster'] == 'testc' and record['command'] == 'echo hi'
        assert record['manifest'] == kubectl.created[0] and record['session']

    def test_wait_journals_the_end_once(self, cluster, kubectl, tmp_path):
        b = Leaf(datalake=str(tmp_path))
        run = cluster.submit('true', block=b)
        kubectl.conditions = ['Complete']
        assert run.wait(poll=0) and run.wait(poll=0)
        assert _events_(b) == ['job:submit', 'job:complete']

    def test_wait_says_what_changes(self, cluster, kubectl, tmp_path, capsys):
        run = cluster.submit('true', block=Leaf(datalake=str(tmp_path)))
        kubectl.conditions = ['Pending']
        run._say_(run.status(), 0, said := {})
        run._say_(run.status(), 60, said)                    # nothing changed: nothing said
        kubectl.conditions = []
        run._say_(run.status(), 120, said)
        kubectl.conditions = ['Complete']
        assert run.wait(poll=0, verbose=True)
        err = capsys.readouterr().err.splitlines()
        assert 'pending after 0.0 min: insufficient quota for cpu' in err[0] and len(err) == 4
        assert 'running after 2.0 min' in err[1] and err[2].endswith('Traceback: boom') and 'complete' in err[3]

    def test_a_failed_job_raises_with_its_log_and_journals_the_error(self, cluster, kubectl, tmp_path):
        b = Leaf(datalake=str(tmp_path))
        run = cluster.submit('true', block=b)
        kubectl.conditions = ['Failed']
        with pytest.raises(RuntimeError, match='boom'):
            run.wait(poll=0)
        assert _events_(b) == ['job:submit', 'job:fail']

    def test_a_job_that_cannot_be_created_journals_the_error(self, cluster, monkeypatch, tmp_path):
        monkeypatch.setattr(kube, '_kubectl_', FakeKubectl(fail_create=True))
        b = Leaf(datalake=str(tmp_path))
        with pytest.raises(RuntimeError, match='forbidden'):
            cluster.submit('true', block=b)
        assert _events_(b) == ['job:submit', 'job:fail']

    def test_status_reads_the_jobs_conditions(self, cluster, kubectl):
        run = cluster.submit('true')
        assert run.status() == 'running'
        kubectl.conditions = ['Complete']
        assert run.status() == 'complete'

    def test_a_name_is_looked_up_among_the_entry_points_each_time(self, entry_points):
        with pytest.raises(KeyError, match="pass the cluster itself"):
            kube.cluster('fromep')
        entry_points.append({'fromep': kube.KubeCluster('fromep', image='i')})
        assert kube.cluster('fromep').name == 'fromep' and list(kube.clusters()) == ['fromep']
        entry_points.clear()
        assert kube.clusters() == {}

    def test_an_entry_point_may_name_one_target_or_a_list(self, entry_points):
        entry_points.extend([kube.KubeCluster('one', image='i'), [kube.KubeCluster('two', image='i'), 'not a target']])
        assert sorted(kube.clusters()) == ['one', 'two']


class TestPipelineRunner:
    def test_the_runs_datalake_is_the_runners_else_the_clusters_else_the_entrys(self, cluster, kubectl, tmp_path,
                                                                                 monkeypatch):
        monkeypatch.setenv('DBX_DATALAKE', str(tmp_path / 'here'))
        p = Single(spec={'n': 5})                                    # its datalake: this machine's
        lake = lambda: _env_(kubectl.created[-1])['DBX_DATALAKE']
        p.runner(cluster).build_tree()
        assert lake() == str(tmp_path / 'here')
        cluster.options = {'env': {'DBX_DATALAKE': '/mnt/team/lake'}}
        p.runner(cluster).build_tree()
        assert lake() == '/mnt/team/lake'
        run = p.runner(cluster, env={'DBX_DATALAKE': '/mnt/team/me/lake'}).build_tree()
        assert lake() == '/mnt/team/me/lake' == run.record['datalake']

    def test_a_pipeline_given_its_datalake_keeps_it_and_its_quote_ships_as_is(self, cluster, kubectl, tmp_path):
        p = Single(spec={'n': 6}, datalake=str(tmp_path))
        cluster.options = {'env': {'DBX_DATALAKE': '/mnt/team/lake'}}
        p.runner(cluster, env={'DBX_DATALAKE': '/mnt/team/me/lake'}).build_tree()
        env = _env_(kubectl.created[-1])
        assert env['DBX_DATALAKE'] == str(tmp_path) and repr(p.quote()) in env['DBX_SUBMIT_EXPRESSION']

    def test_a_pipeline_that_was_not_resolves_the_runs_datalake_there(self, cluster, kubectl, tmp_path, monkeypatch):
        monkeypatch.setenv('DBX_DATALAKE', str(tmp_path))
        p = Single(spec={'n': 8})
        cluster.options = {'env': {'DBX_DATALAKE': '/mnt/team/lake'}}
        p.runner(cluster).build_tree()
        expression = _env_(kubectl.created[-1])['DBX_SUBMIT_EXPRESSION']
        monkeypatch.setenv('DBX_DATALAKE', '/mnt/team/lake')      # as the run's environment has it
        there = dbx.eval(ast.literal_eval(expression.split('dbx.eval(', 1)[1].rsplit('), ', 1)[0]))
        assert there.entry.datalake == '/mnt/team/lake' and there.entry.hash == p.entry.hash

    def test_valid_here_skips_only_when_the_run_builds_here(self, cluster, kubectl, tmp_path, monkeypatch):
        monkeypatch.setenv('DBX_DATALAKE', str(tmp_path))
        p = Single(spec={'n': 7})
        p.entry.build()
        assert p.runner(cluster).build_tree().skipped
        assert not p.runner(cluster, env={'DBX_DATALAKE': '/mnt/team/lake'}).build_tree().skipped

    def test_a_call_ships_the_pipelines_quote_and_the_datalake(self, cluster, kubectl, tmp_path):
        p = Single(spec={'n': 3}, datalake=str(tmp_path), n_gpus=2)
        run = p.runner(cluster).build_tree()
        m = kubectl.created[0]
        expression = _env_(m)['DBX_SUBMIT_EXPRESSION']
        assert expression == f"dbx.cluster.run(dbx.eval({p.quote()!r}), 'build_tree')" == run.record['expression']
        assert run.record['method'] == 'build_tree' and run.record['pipeline_anchor'] == p.anchor
        assert _env_(m)['DBX_DATALAKE'] == str(tmp_path) and _env_(m)['DBX_KUBE_COMMAND'] == 'dbx.pprint --capture-output "$DBX_SUBMIT_EXPRESSION"'
        assert _container_(m)['resources']['requests']['nvidia.com/gpu'] == '2'
        assert dbx.eval(p.quote()).entry.hash == p.entry.hash
        assert _pipe_events_(p) == ['job:submit']

    def test_arguments_travel_as_literals(self, cluster, kubectl, tmp_path):
        p = Single(datalake=str(tmp_path))
        p.runner(cluster, n_gpus=0).build(extra=[1, 'a'])
        assert _env_(kubectl.created[0])['DBX_SUBMIT_EXPRESSION'].endswith("'build', extra=[1, 'a'])")

    def test_a_non_literal_argument_is_refused_before_anything_is_sent(self, cluster, kubectl, tmp_path):
        with pytest.raises(TypeError, match='literals'):
            Single(datalake=str(tmp_path)).runner(cluster).build(object())
        assert kubectl.calls == []

    def test_the_runner_is_of_the_targets_kind_and_refuses_options_it_does_not_take(self, cluster, tmp_path):
        p = Single(datalake=str(tmp_path))
        assert type(p.runner(cluster)) is kube.KubeRunner
        with pytest.raises(TypeError, match='no option partition'):
            p.runner(cluster, partition='gpu')
        with pytest.raises(ValueError, match="'kube', not 'slurm'"):
            p.runner(cluster, mode='slurm')
        assert type(p.runner(cluster, mode='kube')) is kube.KubeRunner

    def test_a_method_the_entry_lacks_is_an_attribute_error(self, cluster, tmp_path):
        with pytest.raises(AttributeError, match='no method'):
            Single(datalake=str(tmp_path)).runner(cluster).launch('no_such_method')

    def test_a_runner_with_no_target_launches_nothing(self, kubectl, tmp_path):
        with pytest.raises(TypeError, match='launches nothing'):
            Single(datalake=str(tmp_path)).runner().build_tree()
        assert kubectl.calls == []

    def test_a_valid_entry_is_not_built_again_unless_forced(self, cluster, kubectl, tmp_path):
        p = Single(datalake=str(tmp_path))
        p.entry.build()
        run = p.runner(cluster).build_tree()
        assert run.job is None and run.skipped and run.status() == 'skipped' and run.wait()
        assert type(run) is kube.KubeRun and 'skipped' in repr(run)
        assert kubectl.calls == []
        assert p.runner(cluster, force=True).build_tree().job is not None


class TestWatching:
    def test_a_call_runs_captured_under_the_submitting_session(self, cluster, kubectl, tmp_path):
        run = Single(datalake=str(tmp_path)).runner(cluster).build_tree()
        env = _env_(kubectl.created[0])
        assert env['DBX_KUBE_COMMAND'] == 'dbx.pprint --capture-output "$DBX_SUBMIT_EXPRESSION"'
        assert env['DBX_SESSION'] == run.record['session'] == str(dbx.BuildJournal.current().session)
        assert env['DBX_RUN'] == run.id and env['DBX_CLUSTER'] == 'testc'
        assert not {'DBX_JOURNAL_SESSION', 'DBX_PARENT_SESSION', 'DBX_JOURNAL_DATALAKE'} & set(env)

    def test_the_runs_output_is_in_the_journal_under_its_session_and_run(self, cluster, tmp_path):
        session = '11111111-2222-3333-4444-555555555555'
        run = kube.KubeRun(cluster, 'job-x', dict(session=session, datalake=str(tmp_path)))
        with pytest.raises(LookupError, match='not over yet'):
            run.command()
        env = {**os.environ, 'DBX_DATALAKE': str(tmp_path), 'DBX_SESSION': session}
        # The submitting command's own row, under the same session, and then the run's.
        subprocess.run([sys.executable, '-c', 'import dbx; dbx.pprint()', "print('the submitter') or 1"],
                       env=env, check=True, capture_output=True)
        subprocess.run([sys.executable, '-c', 'import dbx; dbx.pprint()', '--capture-output',
                        "print('from the job') or 7"], env={**env, 'DBX_RUN': 'job-x', 'DBX_CLUSTER': 'testc'},
                       check=True, capture_output=True)
        assert run.command()['session'] == session and run.command()['run'] == 'job-x'
        assert run.command()['cluster'] == 'testc' 
        assert dict(run.command(datalake=str(tmp_path)))['success'] in (True, 'True')
        assert 'from the job' in _printed_(run.command().output)
        assert 'from the job' in _printed_(run.output)

    def test_a_run_on_a_cluster_with_storage_reads_its_output_there(self, tmp_path):
        session = '11111111-2222-3333-4444-555555555555'
        stored = kube.KubeCluster('s', image='i', storage={'/mnt/team': f"file://{tmp_path}"})
        run = kube.KubeRun(stored, 'job-y', dict(session=session, datalake='/mnt/team/lake'))
        env = {**os.environ, 'DBX_DATALAKE': str(tmp_path / 'lake'), 'DBX_SESSION': session, 'DBX_RUN': 'job-y'}
        subprocess.run([sys.executable, '-c', 'import dbx; dbx.pprint()', '--capture-output',
                        "print('on its own disk') or 1"], env=env, check=True, capture_output=True)
        assert 'on its own disk' in _printed_(run.output)

    def test_the_output_is_read_where_the_journal_is_read_not_where_it_was_written(self, cluster, tmp_path):
        import shutil
        there, here = tmp_path / 'there', tmp_path / 'here'
        # A record from before 2026-11-04: the run had a session of its own, run_session.
        run = kube.KubeRun(cluster, 'job-x', dict(run_session='11111111-2222-3333-4444-555555555555',
                                                  datalake=str(there)))
        env = {**os.environ, 'DBX_DATALAKE': str(there), 'DBX_SESSION': run.record['run_session']}
        subprocess.run([sys.executable, '-c', 'import dbx; dbx.pprint()', '--capture-output',
                        "print('written there') or 1"], env=env, check=True, capture_output=True)
        shutil.copytree(there, here)
        shutil.rmtree(there)   # What the row records is gone: the copy is all there is.
        row = run.command(datalake=str(here))
        assert str(there) in row['output_captures'][0]
        assert 'written there' in _printed_(row.output)
        assert 'written there' in _printed_(lambda: row.output(basename='master'))

    def test_the_runs_of_a_block_come_back_from_its_journal(self, cluster, kubectl, tmp_path):
        p = Single(datalake=str(tmp_path))
        first = p.runner(cluster).build_tree()
        kubectl.conditions = ['Complete']
        first.wait(poll=0)
        second = p.runner(cluster).build_tree()
        runs = dbx.cluster.runs(p)
        assert [r.id for r in runs] == [first.id, second.id] and type(runs[0]) is kube.KubeRun
        assert runs[0]._closed_ and not runs[1]._closed_
        assert runs[1].record['manifest'] == kubectl.created[1]
        runs[1].wait(poll=0)
        assert _pipe_events_(p) == ['job:submit', 'job:complete', 'job:submit', 'job:complete']

    def test_a_runs_target_comes_back_from_its_record_not_by_its_name(self, cluster, kubectl, entry_points, tmp_path):
        p = Single(datalake=str(tmp_path))
        p.runner(cluster).build_tree()
        assert kube.clusters() == {}            # nothing to look 'testc' up in
        (run,) = dbx.cluster.runs(p)
        assert run.target == cluster and run.target is not cluster

    def test_a_name_submits_to_the_entry_points_target(self, cluster, kubectl, entry_points, tmp_path):
        entry_points.append({'testc': cluster})
        run = Single(datalake=str(tmp_path)).runner('testc').build_tree()
        assert run.record['target_spec']['fields']['queue'] == 'gpu-q' and kubectl.created

    def test_a_pending_job_says_why(self, cluster, kubectl):
        run = cluster.submit('true')
        kubectl.conditions = ['Pending']
        assert run.status() == 'pending' and run.reason() == 'insufficient quota for cpu'
        df = kube.jobs([cluster])
        assert list(df['job']) == [run.id] and df['reason'].iloc[0] == 'insufficient quota for cpu'
        assert df['command'].iloc[0] == 'true'
        assert list(cluster.jobs()['job']) == [run.id]      # the one cluster's, by its method

    def test_jobs_are_the_users_unless_all_users(self, cluster, kubectl, monkeypatch):
        cluster.submit('true')
        cluster.user = 'someone-else'
        assert len(kube.jobs([cluster])) == 0 and len(kube.jobs([cluster], all_users=True)) == 1


class TestRunner:
    """A Runner's runs: scoped by hash, by target, or (by class or anchor) not at all; listed, tabled and cancelled."""

    @pytest.fixture
    def other(self, cluster):
        return kube.KubeCluster('otherc', context='ctx2', namespace='ns', queue='q', image='img:1', user='me')

    def test_every_submit_entry_names_its_cluster_and_run(self, cluster, kubectl, tmp_path):
        p = Single(datalake=str(tmp_path))
        run = p.runner(cluster).build_tree()
        kubectl.conditions = ['Complete']
        run.wait(poll=0)
        j = PipeJournal.read(f"{PipeJournal.ANCHOR}/{p.anchor}", datalake=str(tmp_path), index=None, event='^job:')
        assert list(j['event']) != [] and set(j['cluster']) == {'testc'} and set(j['run']) == {run.id}
        assert set(j['entrypath']) == {p.entry.anchorkeypath}
        assert _events_(p.entry) == []      # nothing on the entry: the pipeline launched it
        env = _env_(kubectl.created[0])
        assert env['DBX_CLUSTER'] == 'testc' and env['DBX_RUN'] == run.id

    def test_entries_written_in_a_run_name_it_and_others_do_not(self, monkeypatch, tmp_path):
        dblk = Leaf(datalake=str(tmp_path))
        dblk.build()
        monkeypatch.setenv('DBX_CLUSTER', 'testc')
        monkeypatch.setenv('DBX_RUN', 'me-leaf-abcde')
        Leaf(spec={'n': 2}, datalake=str(tmp_path)).build()
        j = Leaf(datalake=str(tmp_path)).buildjournal(event='build:end')
        by_hash = {h: (c, r) for h, c, r in zip(j['hash'], j['cluster'], j['run'])}
        assert by_hash[Leaf(spec={'n': 2}, datalake=str(tmp_path)).hash] == ('testc', 'me-leaf-abcde')
        assert by_hash[dblk.hash] == (None, None)

    def test_runs_are_the_datablocks_own_not_its_classs(self, cluster, kubectl, tmp_path):
        one, two = Single(spec={'n': 1}, datalake=str(tmp_path)), Single(spec={'n': 2}, datalake=str(tmp_path))
        r1, r2 = one.runner(cluster).build_tree(), two.runner(cluster).build_tree()
        assert [r.id for r in one.runner().runs()] == [r1.id]
        assert [r.id for r in dbx.cluster.runs(two)] == [r2.id]
        assert [r.id for r in dbx.cluster.runs(two.entry)] == [r2.id]
        everyone = [r1.id, r2.id]
        assert [r.id for r in dbx.runner(Leaf, datalake=str(tmp_path)).runs()] == everyone
        assert [r.id for r in dbx.runner(two.entry.anchor, datalake=str(tmp_path)).runs()] == everyone

    def test_runs_on_a_target_are_those_on_it(self, cluster, other, kubectl, tmp_path):
        p = Single(datalake=str(tmp_path))
        here, there = p.runner(cluster).build_tree(), p.runner(other).build_tree()
        assert [r.id for r in p.runner(cluster).runs()] == [here.id]
        assert [r.id for r in p.runner(other).runs()] == [there.id]
        assert [r.id for r in p.runner().runs()] == [here.id, there.id]
        assert type(p.runner().runs()[1].target) is kube.KubeCluster and p.runner().runs()[1].target.name == 'otherc'

    def test_open_and_closed(self, cluster, kubectl, tmp_path):
        p = Single(datalake=str(tmp_path))
        done = p.runner(cluster).build_tree()
        kubectl.conditions = ['Complete']
        done.wait(poll=0)
        kubectl.conditions = []
        still = p.runner(cluster).build_tree()
        r = p.runner(cluster)
        assert [x.id for x in r.runs(open=True)] == [still.id] and [x.id for x in r.runs(open=False)] == [done.id]
        assert r.runs(open=False)[0].closing[0] == 'job:complete'

    def test_a_close_journaled_before_entries_named_their_run_is_found_by_its_note(self, cluster, kubectl, tmp_path):
        p = Single(datalake=str(tmp_path))
        run = p.runner(cluster).build_tree()
        other = p.runner(cluster).build_tree()
        p.entry.write_journal_entry(event='submit:END', note=f"kube:testc job/{run.id} complete", inline_note=True)
        assert [x.id for x in p.runner().runs(open=True)] == [other.id]

    def test_cancel_cancels_what_is_pending_or_running_and_journals_it(self, cluster, kubectl, tmp_path):
        one, two = Single(spec={'n': 1}, datalake=str(tmp_path)), Single(spec={'n': 2}, datalake=str(tmp_path))
        r1, r2 = one.runner(cluster).build_tree(), two.runner(cluster).build_tree()
        assert [r.id for r in one.runner(cluster).cancel()] == [r1.id]
        assert kubectl.deleted == {r1.id}
        assert _pipe_events_(one) == ['job:submit', 'job:cancel'] and _pipe_events_(two)[-1] == 'job:submit'
        assert one.runner(cluster).cancel() == []

    def test_cancel_by_class_journals_on_each_runs_own_datablock(self, cluster, kubectl, tmp_path, monkeypatch):
        monkeypatch.setenv('DBX_DATALAKE', str(tmp_path))
        one, two = Single(spec={'n': 1}, datalake=str(tmp_path)), Single(spec={'n': 2}, datalake=str(tmp_path))
        r1, r2 = one.runner(cluster).build_tree(), two.runner(cluster).build_tree()
        assert sorted(r.id for r in dbx.runner(Leaf, datalake=str(tmp_path)).cancel()) == sorted([r1.id, r2.id])
        for p in (one, two):
            assert _pipe_events_(p) == ['job:submit', 'job:cancel']

    def test_a_job_that_is_over_or_gone_is_not_cancelled(self, cluster, kubectl, tmp_path):
        p = Single(datalake=str(tmp_path))
        run = p.runner(cluster).build_tree()
        kubectl.deleted.add(run.id)
        assert p.runner(cluster).cancel() == [] and _pipe_events_(p) == ['job:submit']

    def test_dash(self, cluster, kubectl, tmp_path):
        p = Single(datalake=str(tmp_path))
        done = p.runner(cluster).build_tree()
        kubectl.conditions = ['Complete']
        done.wait(poll=0)
        kubectl.conditions = []
        cancelled = p.runner(cluster).build_tree()
        cancelled.cancel()
        running = p.runner(cluster).launch('build')
        df = p.runner(cluster).runs().dash()
        assert list(df.columns) == ['run', 'cluster', 'entry', 'method', 'status', 'submitted', 'started', 'ended',
                                    'observed', 'elapsed', 'outcome', 'event']
        assert list(df['run']) == [done.id, cancelled.id, running.id]
        assert list(df['status']) == ['complete', 'cancelled', 'running']
        assert list(df['method']) == ['build_tree', 'build_tree', 'build']
        assert list(df['event'].fillna('')) == ['job:complete', 'job:cancel', '']
        assert set(df['entry']) == {p.entry.key} and set(df['cluster']) == {'testc'}
        assert df['elapsed'].notna().all() and df['observed'].isna().tolist() == [False, False, True]
        assert df['started'].isna().all() and df['outcome'].isna().all()     # no run journaled its own here
        assert list(p.runner(cluster).runs().dash(live=False)['status']) == ['complete', 'cancelled', 'open']
        assert list(dbx.cluster.runs(Leaf, datalake=str(tmp_path)).dash()['run']) == list(df['run'])

    def test_runs_print_as_their_dash_each_row_its_index(self, cluster, kubectl, tmp_path):
        p = Single(datalake=str(tmp_path))
        first = p.runner(cluster).build_tree()
        first.cancel()
        last = p.runner(cluster).launch('build')
        rs = p.runner(cluster).runs()
        lines = repr(rs).splitlines()
        assert lines[0].split() == ['run', 'cluster', 'method', 'status', 'submitted', 'elapsed']
        assert lines[1].split()[:5] == ['0', first.id, 'testc', 'build_tree', 'cancelled']
        assert lines[2].split()[:5] == ['1', last.id, 'testc', 'build', 'running']
        assert rs[-1].id == last.id and isinstance(rs[:1], dbx.cluster.Runs)
        assert _pipe_events_(p) == ['job:submit', 'job:cancel', 'job:submit']    # printing journals nothing
        assert repr(dbx.cluster.Runs()) == 'Runs([])'

    def test_runs_printed_journal_nothing_even_for_a_run_found_over(self, cluster, kubectl, tmp_path):
        p = Single(datalake=str(tmp_path))
        p.runner(cluster).build_tree()
        kubectl.conditions = ['Complete']
        assert 'complete' in repr(p.runner().runs())
        assert _pipe_events_(p) == ['job:submit']

    def test_a_runs_repr_is_what_is_journaled(self, cluster, kubectl, tmp_path):
        p = Single(datalake=str(tmp_path))
        run = p.runner(cluster).build_tree()
        run.cancel()
        r = repr(p.runner(cluster).runs()[0])
        assert r.startswith(f"KubeRun(testc, {run.id}, build_tree, submitted ") and r.endswith(', cancelled)')

    def test_cancel_a_row_by_its_index(self, cluster, kubectl, tmp_path):
        p = Single(datalake=str(tmp_path))
        keep = p.runner(cluster).build_tree()
        gone = p.runner(cluster).build_tree()
        assert [r.id for r in p.runner(cluster).runs()[-1:].cancel()] == [gone.id]
        assert [r.id for r in p.runner(cluster).runs(open=True)] == [keep.id]

    def test_a_datablock_built_in_a_run_finds_the_run_through_its_build_entries(self, cluster, kubectl, tmp_path,
                                                                                  monkeypatch):
        p = Single(spec={'n': 1}, datalake=str(tmp_path))
        run = p.runner(cluster).build_tree()
        built_there = Leaf(spec={'n': 5}, datalake=str(tmp_path))   # in no pipeline's entry: built inside the Job
        monkeypatch.setenv('DBX_CLUSTER', 'testc')
        monkeypatch.setenv('DBX_RUN', run.id)
        built_there.build()
        monkeypatch.delenv('DBX_RUN')
        assert [r.id for r in dbx.runner(built_there).runs()] == [run.id]
        assert dbx.runner(Leaf(spec={'n': 6}, datalake=str(tmp_path))).runs() == []

    def test_a_pipeline_class_lists_the_runs_of_every_pipeline_of_it(self, cluster, kubectl, tmp_path):
        one, two = Single(spec={'n': 1}, datalake=str(tmp_path)), Single(spec={'n': 2}, datalake=str(tmp_path))
        r1, r2 = one.runner(cluster).build_tree(), two.runner(cluster).build_tree()
        assert [r.id for r in dbx.runner(Single, datalake=str(tmp_path)).runs()] == [r1.id, r2.id]

    def test_runs_journaled_on_the_entry_before_are_still_found(self, cluster, kubectl, tmp_path):
        p = Single(datalake=str(tmp_path))
        old = cluster.submit('true', block=p.entry)    # the layout of before: submit:* on the entry
        new = p.runner(cluster).build_tree()
        assert [r.id for r in p.runner().runs()] == [old.id, new.id]
        assert [r.id for r in dbx.runner(Leaf, datalake=str(tmp_path)).runs()] == [old.id, new.id]
        assert [r.id for r in p.runner().cancel()] == [old.id, new.id]
        assert _events_(p.entry) == ['job:submit', 'job:cancel']
        assert _pipe_events_(p) == ['job:submit', 'job:cancel']

    def test_the_pipeline_journal_still_reads_and_compacts_as_creations(self, cluster, kubectl, tmp_path,
                                                                        monkeypatch):
        monkeypatch.setenv('DBX_DATALAKE', str(tmp_path))
        monkeypatch.setattr(Single, 'JOURNAL', True)
        p = Single(datalake=str(tmp_path))
        p.entry
        p.runner(cluster).build_tree()
        Single(datalake=str(tmp_path)).entry                       # a second creation of the same pipeline
        run = p.runner(cluster).build_tree()
        (row,) = dbx.pipejournal(datalake=str(tmp_path)).to_dict('records')
        assert row['event'] == 'create' and row['entrypaths'] == [p.entry.anchorkeypath]
        PipeJournal.UNSAFE_compact(datalake=str(tmp_path), OVERRIDE=True)
        assert len(p.runner().runs()) == 2 and p.runner().runs()[-1].id == run.id
        assert len(dbx.pipejournal(datalake=str(tmp_path))) == 1
        monkeypatch.setattr(Single, 'JOURNAL', False)
        PipeJournal.UNSAFE_compact(datalake=str(tmp_path), unjournaled=True, OVERRIDE=True)
        assert len(dbx.pipejournal(datalake=str(tmp_path))) == 0 and len(p.runner().runs()) == 2

    def test_the_run_journals_its_own_start_and_end_where_it_runs(self, cluster, kubectl, tmp_path, monkeypatch):
        p = Single(datalake=str(tmp_path))
        launched = p.runner(cluster).build_tree()
        monkeypatch.setenv('DBX_CLUSTER', 'testc')
        monkeypatch.setenv('DBX_RUN', launched.id)
        dbx.cluster.run(dbx.eval(p.quote()), 'build_tree')               # what the Job evaluates
        assert p.entry.valid()
        j = PipeJournal.read(f"{PipeJournal.ANCHOR}/{p.anchor}", datalake=str(tmp_path), index=None, event='^job:')
        mine = j[j['run'] == launched.id].sort_values('datetime')
        assert list(mine['event']) == ['job:submit', 'job:start', 'job:end'] and mine['note'].iloc[-1] == 'ok'
        df = p.runner().runs().dash(live=False)
        assert df['outcome'].iloc[0] == 'ok' and df['started'].notna().all() and df['ended'].notna().all()
        assert df['status'].iloc[0] == 'open'           # the run says it ended; the submitter has not seen it

    def test_a_failing_run_says_what_it_raised(self, tmp_path, monkeypatch):
        monkeypatch.setenv('DBX_RUN', 'job-y')
        with pytest.raises(AttributeError):
            dbx.cluster.run(Single(datalake=str(tmp_path)), 'no_such_method')
        j = PipeJournal.read(f"{PipeJournal.ANCHOR}/{Single().anchor}", datalake=str(tmp_path), index=None,
                             event='job:end')
        assert j['note'].iloc[0].startswith('AttributeError')

    def test_dash_closes_a_run_it_finds_over(self, cluster, kubectl, tmp_path):
        p = Single(datalake=str(tmp_path))
        run = p.runner(cluster).build_tree()
        kubectl.conditions = ['Complete']
        assert list(p.runner().runs().dash()['status']) == ['complete']
        assert _pipe_events_(p) == ['job:submit', 'job:complete']
        p.runner().runs().dash()                                            # closed once, not twice
        assert _pipe_events_(p) == ['job:submit', 'job:complete'] and p.runner().runs(open=False)[0].id == run.id

    def test_runs_journaled_under_the_old_event_names_are_read(self, cluster, kubectl, tmp_path):
        p = Single(datalake=str(tmp_path))
        run = p.runner(cluster).build_tree()
        from dbx.journals import PipeJournal as PJ
        PJ.write(p, 'submit:END', entrypath=p.entrypath, note=f"kube:testc job/{run.id} complete", inline_note=True,
                 cluster='testc', run=run.id)
        (closed,) = p.runner().runs(open=False)
        assert closed.id == run.id and closed.closing[0] == 'job:complete'

    def test_adopted_runs_are_those_of_the_builds_a_redirection_reads(self, tmp_path):
        import pandas as pd
        j = pd.DataFrame({'hash': ['mine', 'theirs', 'others'],
                          'anchorkeypath': ['/lake/A/mine', '/lake/A/theirs', '/lake/A/others']})

        class Dblk:
            hash = 'mine'
            _redirected_paths_ = {'x': '/lake/A/theirs/x.txt'}
        assert list(dbxcluster._of_dblk_(j, Dblk, adopted=False)) == [True, False, False]
        assert list(dbxcluster._of_dblk_(j, Dblk, adopted=True)) == [True, True, False]


class TestClusterDryRun:
    def test_it_ships_a_small_run_that_verifies_the_entry_and_records_the_call(self, cluster, kubectl, tmp_path):
        p = Single(spec={'n': 2}, datalake=str(tmp_path), n_gpus=8)
        p.entry.build()
        run = p.runner(cluster, dry_run='cluster').build_tree()
        m = kubectl.created[0]
        assert run.id is not None and m['metadata']['name'].startswith('me-single-build-tree-check-')
        assert _env_(m)['DBX_SUBMIT_EXPRESSION'] == f"dbx.cluster.verify(dbx.eval({p.quote()!r}).entry, {p.entry.hash!r})"
        assert _env_(m)['DBX_SUBMIT_CALL'].endswith("'build_tree')") and run.record['method'] == 'verify'
        req = _container_(m)['resources']['requests']
        assert (req['nvidia.com/gpu'], req['cpu']) == ('0', '2') and 'ray start' not in _container_(m)['command'][2]

    def test_verify_reports_the_entry_and_refuses_another_hash(self, tmp_path):
        b = Leaf(datalake=str(tmp_path))
        got = dbx.cluster.verify(dbx.eval(b.quote()), b.hash)
        assert got['hash'] == b.hash and got['valid'] is False
        with pytest.raises(ValueError, match='differs'):
            dbx.cluster.verify(b, '0' * 8)


# --------------------------------------------------------------------------------------------- #
# queues
# --------------------------------------------------------------------------------------------- #

def _queue_(name, quota, used, *, cohort='co', pending=0, admitted=0, resource='nvidia.com/gpu'):
    return {'metadata': {'name': name},
            'spec': {'cohortName': cohort, 'resourceGroups': [{'flavors': [
                {'resources': [{'name': resource, 'nominalQuota': quota}, {'name': 'cpu', 'nominalQuota': '500m'}]}]}]},
            'status': {'flavorsUsage': [{'resources': [{'name': resource, 'total': used, 'borrowed': 0}]}],
                       'pendingWorkloads': pending, 'admittedWorkloads': admitted}}


class TestQueues:
    @pytest.fixture
    def queues(self, monkeypatch, entry_points):
        entry_points.append({n: kube.KubeCluster(n, image='i') for n in ('a', 'b', 'gone')})
        answers = {'a': [_queue_('team-q', '24', '20', admitted=2), _queue_('idle-q', '0', '0'),
                         _queue_('mem-q', '187Gi', '93.5Gi', resource='memory')],
                   'b': [_queue_('team-q', '48', '32', pending=1)]}

        def fake(context, *args, input=None):
            if context not in answers:
                raise RuntimeError(f"Unable to connect to the server: {context}")
            return json.dumps({'items': answers[context]})
        monkeypatch.setattr(kube, '_kubectl_', fake)

    def test_one_row_per_queue_with_quota_or_use(self, queues):
        df = kube.queues(['a', 'b'])
        assert list(df['queue']) == ['team-q', 'team-q']
        row = df[df['cluster'] == 'b'].iloc[0]
        assert (row['quota'], row['used'], row['free'], row['pending']) == (48, 32, 16, 1)

    def test_all_and_a_queue_filter(self, queues):
        assert 'idle-q' in set(kube.queues(['a'], all=True)['queue'])
        assert list(kube.queues(['a', 'b'], queues='^team')['cluster']) == ['a', 'b']

    def test_memory_is_in_gib_and_cpu_in_cores(self, queues):
        assert kube.queues(['a'], resource='memory', queues='mem')['used'].iloc[0] == 93.5
        assert kube.queues(['a'], resource='cpu', queues='team')['quota'].iloc[0] == 0.5

    def test_an_unreachable_cluster_is_a_row_with_its_error(self, queues):
        df = kube.queues(['a', 'gone'])
        assert 'Unable to connect' in df[df['cluster'] == 'gone']['error'].iloc[0]

    def test_one_clusters_queues_by_its_method_raising_when_unreachable(self, queues):
        assert list(kube.cluster('b').queues()['quota']) == [48]
        with pytest.raises(RuntimeError, match='Unable to connect'):
            kube.cluster('gone').queues()

    def test_the_console_script_prints_the_table(self, queues, capsys):
        assert kube.queues_main(['b', '-q', 'team']) is None
        assert 'team-q' in capsys.readouterr().out

    def test_by_default_every_entry_point_cluster_and_a_shared_context_once(self, queues, entry_points):
        entry_points.append([kube.KubeCluster('a-cpu', context='a', image='i')])
        df = kube.queues()
        assert sorted(df['cluster']) == ['a', 'b', 'gone'] and list(df['queue'].dropna()) == ['team-q', 'team-q']


def _sftp_options_():
    return {'username': 'from-a-function'}
