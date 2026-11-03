"""dbx.kube: the Job a command or a pipeline's call becomes, its submission journaled, and kubedash.

kubectl is never run: `dbx.kube._kubectl_` is replaced by a fake that records what it is handed and
answers as the cluster would.
"""
import json
import os
import subprocess
import sys
from dataclasses import dataclass

import pytest

import dbx
from dbx import kube, submits
from dbx.datablocks import DATAFILE, Datablock
from dbx.pipelines import Pipeline


@pytest.fixture
def cluster():
    c = kube.register(kube.Cluster('testc', context='ctx', namespace='ns', queue='gpu-q', cpu_queue='cpu-q',
                                   priority='normal', mounts={'/mnt/team': 'team-pvc', '/mnt/data': 'data-pvc'},
                                   readonly=('/mnt/data',), cpu_per_gpu=10, gib_per_gpu=100, image='img:1',
                                   user='me'))
    yield c
    submits.TARGETS.pop('testc', None)


class FakeKubectl:
    """What kubectl was asked, and the Job status it reports."""

    def __init__(self, conditions=(), fail_create=False):
        self.calls, self.created, self.conditions, self.fail_create = [], [], list(conditions), fail_create

    def __call__(self, context, *args, input=None):
        self.calls.append((context, args))
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
        m = kube.job('echo hi', 'testc', gpus=2, name='My Thing')
        res = _container_(m)['resources']['requests']
        assert (res['cpu'], res['memory'], res['nvidia.com/gpu']) == ('20', '200Gi', '2')
        assert m['metadata']['labels']['kueue.x-k8s.io/queue-name'] == 'gpu-q'
        assert m['metadata']['labels']['kueue.x-k8s.io/priority-class'] == 'normal'
        assert m['metadata']['name'].startswith('me-my-thing-') and m['metadata']['namespace'] == 'ns'
        assert m['spec']['suspend'] is True and m['spec']['backoffLimit'] == 0
        assert _env_(m)['DBX_KUBE_COMMAND'] == 'echo hi' and m['metadata']['annotations']['dbx/command'] == 'echo hi'

    def test_a_job_without_gpus_goes_to_the_cpu_queue(self, cluster):
        m = kube.job('echo hi', 'testc')
        assert m['metadata']['labels']['kueue.x-k8s.io/queue-name'] == 'cpu-q'
        assert _container_(m)['resources']['requests']['nvidia.com/gpu'] == '0'

    def test_the_claims_mount_where_the_cluster_says_read_only_where_it_says(self, cluster):
        m = kube.job('true', 'testc')
        mounts = {v['mountPath']: v for v in _container_(m)['volumeMounts']}
        claims = {v['name']: v for v in m['spec']['template']['spec']['volumes']}
        assert claims[mounts['/mnt/team']['name']]['persistentVolumeClaim']['claimName'] == 'team-pvc'
        assert mounts['/mnt/data'].get('readOnly') and not mounts['/mnt/team'].get('readOnly')

    def test_off_anyscale_nothing_of_anyscales_is_assumed(self, cluster):
        m = kube.job('true', 'testc', ray=True)
        script = _container_(m)['command'][2]
        assert 'securityContext' not in m['spec']['template']['spec']
        assert not any(k.startswith('ANYSCALE') for k in _env_(m))
        assert '\nray start --head' in script and '/home/ray' not in script

    def test_anyscale_adds_its_user_its_ray_and_its_working_dir(self, cluster):
        m = kube.job('true', 'testc', anyscale=True, ray=True)
        script = _container_(m)['command'][2]
        assert m['spec']['template']['spec']['securityContext'] == {'runAsUser': 1000, 'runAsGroup': 100}
        assert _env_(m)['ANYSCALE_WORKING_DIR'] == dbx.anyscale.WORKING_DIR
        assert '/home/ray/venv/bin/ray start --head' in script and 'mkdir -p "$ANYSCALE_WORKING_DIR"' in script

    def test_no_image_is_an_error_that_says_where_one_comes_from(self, cluster):
        cluster.image = None
        with pytest.raises(ValueError, match='image='):
            kube.job('true', 'testc')

    def test_the_venv_is_activated_and_the_command_runs_last(self, cluster):
        script = _container_(kube.job('true', 'testc', venv='/mnt/team/venv', setup=['echo setup']))['command'][2]
        lines = script.splitlines()
        assert lines.index('echo setup') < lines.index('source /mnt/team/venv/bin/activate')
        assert lines[-1] == 'exec bash -c "$DBX_KUBE_COMMAND"'

    def test_the_clusters_options_are_defaults_and_its_env_is_added_to(self, cluster):
        cluster.options = {'venv': '/v', 'env': {'A': '1'}, 'gpus': 1}
        m = kube.job('true', 'testc', env={'B': '2'}, gpus=4)
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
        m = kube.job('true', 'testc', revision='HEAD:HEAD')
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
            kube.job('true', 'testc', revision=True)


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

    def test_sync_copies_what_changed_keeps_modes_and_links(self, tmp_path):
        src, dst = tmp_path / 'src', tmp_path / 'dst'
        (src / 'bin').mkdir(parents=True)
        (src / 'bin' / 'tool').write_text('#!/bin/sh\n')
        os.chmod(src / 'bin' / 'tool', 0o755)
        (src / 'data.txt').write_text('abc')
        os.symlink('data.txt', src / 'link')
        first = kube.sync(str(src), f"file://{dst}")
        assert first['files'] == 2 and os.stat(dst / 'bin' / 'tool').st_mode & 0o777 == 0o755
        assert os.readlink(dst / 'link') == 'data.txt'
        assert kube.sync(str(src), f"file://{dst}") == dict(files=0, bytes=0, skipped=2)
        (src / 'data.txt').write_text('abcd')
        assert kube.sync(str(src), f"file://{dst}")['files'] == 1 and (dst / 'data.txt').read_text() == 'abcd'

    def test_a_cluster_that_mounts_our_paths_takes_a_workdir_only_where_it_is(self, cluster, kubectl, tmp_path):
        kube.submit('true', 'testc', workdirs={str(tmp_path): str(tmp_path)})
        with pytest.raises(ValueError, match='there already'):
            kube.submit('true', 'testc', workdirs={str(tmp_path): '/mnt/team/x'})

    def test_a_cluster_with_its_own_storage_is_synced_before_the_job(self, cluster, kubectl, tmp_path):
        (tmp_path / 'src').mkdir()
        (tmp_path / 'src' / 'f').write_text('x')
        cluster.storage = {'/mnt/team': f"file://{tmp_path / 'remote'}"}
        run = kube.submit('true', 'testc', workdirs={str(tmp_path / 'src'): '/mnt/team/code'})
        assert (tmp_path / 'remote' / 'code' / 'f').read_text() == 'x'
        assert run.record['workdirs'][str(tmp_path / 'src')]['files'] == 1


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


def _events_(block):
    try:
        j = block.buildjournal()
    except FileNotFoundError:  # No journal at all: nothing was ever written for it.
        return []
    return [] if j is None or len(j) == 0 else list(j.sort_values('datetime')['event'])


class TestSubmit:
    def test_a_dry_run_contacts_nothing_and_journals_nothing(self, cluster, kubectl, tmp_path):
        b = Leaf(datalake=str(tmp_path))
        run = kube.submit('true', 'testc', block=b, dry_run=True)
        assert kubectl.calls == [] and run.job is None and run.status() == 'dry-run'
        assert _events_(b) == [] and run.record['manifest']['kind'] == 'Job'

    def test_a_submission_is_journaled_with_its_whole_record(self, cluster, kubectl, tmp_path):
        b = Leaf(datalake=str(tmp_path))
        run = kube.submit('echo hi', 'testc', block=b, gpus=1)
        assert kubectl.created[0]['metadata']['name'] == run.job
        j = b.buildjournal()
        [row] = j[j['event'] == 'submit:BEGIN'].to_dict('records')
        with open(row['note']) as f:
            record = json.load(f)
        assert record['job'] == run.job and record['cluster'] == 'testc' and record['command'] == 'echo hi'
        assert record['manifest'] == kubectl.created[0] and record['session']

    def test_wait_journals_the_end_once(self, cluster, kubectl, tmp_path):
        b = Leaf(datalake=str(tmp_path))
        run = kube.submit('true', 'testc', block=b)
        kubectl.conditions = ['Complete']
        assert run.wait(poll=0) and run.wait(poll=0)
        assert _events_(b) == ['submit:BEGIN', 'submit:END']

    def test_a_failed_job_raises_with_its_log_and_journals_the_error(self, cluster, kubectl, tmp_path):
        b = Leaf(datalake=str(tmp_path))
        run = kube.submit('true', 'testc', block=b)
        kubectl.conditions = ['Failed']
        with pytest.raises(RuntimeError, match='boom'):
            run.wait(poll=0)
        assert _events_(b) == ['submit:BEGIN', 'submit:ERROR']

    def test_a_job_that_cannot_be_created_journals_the_error(self, cluster, monkeypatch, tmp_path):
        monkeypatch.setattr(kube, '_kubectl_', FakeKubectl(fail_create=True))
        b = Leaf(datalake=str(tmp_path))
        with pytest.raises(RuntimeError, match='forbidden'):
            kube.submit('true', 'testc', block=b)
        assert _events_(b) == ['submit:BEGIN', 'submit:ERROR']

    def test_status_reads_the_jobs_conditions(self, cluster, kubectl):
        run = kube.submit('true', 'testc')
        assert run.status() == 'running'
        kubectl.conditions = ['Complete']
        assert run.status() == 'complete'

    def test_an_unregistered_cluster_says_how_to_register_one(self):
        with pytest.raises(KeyError, match='register'):
            kube.cluster('nowhere')

    def test_a_target_is_found_through_its_packages_entry_point(self, monkeypatch):
        class EP:
            def load(self):
                kube.register(kube.Cluster('fromep', image='i'))
        monkeypatch.setattr(submits, '_LOADED_', False)
        monkeypatch.setattr(submits.metadata, 'entry_points', lambda group: [EP()] if group == 'dbx.targets' else [])
        try:
            assert kube.cluster('fromep').name == 'fromep' and 'fromep' in kube.clusters()
        finally:
            submits.TARGETS.pop('fromep', None)


class TestPipelineSubmit:
    def test_a_call_ships_the_pipelines_quote_and_the_datalake(self, cluster, kubectl, tmp_path):
        p = Single(spec={'n': 3}, datalake=str(tmp_path), n_gpus=2)
        run = p.submit('testc').build_tree()
        m = kubectl.created[0]
        expression = _env_(m)['DBX_SUBMIT_EXPRESSION']
        assert expression == f"dbx.eval({p.quote()!r}).entry.build_tree()" == run.record['expression']
        assert _env_(m)['DBX_DATALAKE'] == str(tmp_path) and _env_(m)['DBX_KUBE_COMMAND'] == 'dbx.pprint --capture-output "$DBX_SUBMIT_EXPRESSION"'
        assert _container_(m)['resources']['requests']['nvidia.com/gpu'] == '2'
        assert dbx.eval(p.quote()).entry.hash == p.entry.hash
        assert _events_(p.entry) == ['submit:BEGIN']

    def test_entry_reads_as_the_call_and_arguments_travel_as_literals(self, cluster, kubectl, tmp_path):
        p = Single(datalake=str(tmp_path))
        p.submit('testc', gpus=0).entry.build(extra=[1, 'a'])
        assert _env_(kubectl.created[0])['DBX_SUBMIT_EXPRESSION'].endswith(".entry.build(extra=[1, 'a'])")

    def test_a_non_literal_argument_is_refused_before_anything_is_sent(self, cluster, kubectl, tmp_path):
        with pytest.raises(TypeError, match='literals'):
            Single(datalake=str(tmp_path)).submit('testc').build(object())
        assert kubectl.calls == []

    def test_the_submit_is_of_the_targets_kind_and_refuses_options_it_does_not_take(self, cluster, tmp_path):
        p = Single(datalake=str(tmp_path))
        assert type(p.submit('testc')) is kube.KubeSubmit and type(p.submit(cluster)) is kube.KubeSubmit
        with pytest.raises(TypeError, match='no option partition'):
            p.submit('testc', partition='gpu')
        with pytest.raises(ValueError, match="'kube', not 'slurm'"):
            p.submit('testc', mode='slurm')
        assert type(p.submit('testc', mode='kube')) is kube.KubeSubmit

    def test_a_method_the_entry_lacks_is_an_attribute_error(self, cluster, tmp_path):
        with pytest.raises(AttributeError, match='no method'):
            Single(datalake=str(tmp_path)).submit('testc').no_such_method

    def test_a_valid_entry_is_not_built_again_unless_forced(self, cluster, kubectl, tmp_path):
        p = Single(datalake=str(tmp_path))
        p.entry.build()
        run = p.submit('testc').build_tree()
        assert run.job is None and run.skipped and run.status() == 'skipped' and run.wait()
        assert type(run) is kube.KubeRun and 'skipped' in repr(run)
        assert kubectl.calls == []
        assert p.submit('testc', force=True).build_tree().job is not None


class TestWatching:
    def test_a_call_runs_captured_under_a_session_of_its_own_that_is_recorded(self, cluster, kubectl, tmp_path):
        run = Single(datalake=str(tmp_path)).submit('testc').build_tree()
        env = _env_(kubectl.created[0])
        assert env['DBX_KUBE_COMMAND'] == 'dbx.pprint --capture-output "$DBX_SUBMIT_EXPRESSION"'
        assert env['DBX_JOURNAL_SESSION'] == run.record['run_session'] != env['DBX_PARENT_SESSION']
        assert env['DBX_JOURNAL_DATALAKE'] == str(tmp_path)

    def test_the_runs_output_is_in_the_journal_under_its_session(self, cluster, tmp_path):
        run = kube.KubeRun(cluster, 'job-x', dict(run_session='11111111-2222-3333-4444-555555555555',
                                                  datalake=str(tmp_path)))
        with pytest.raises(LookupError, match='not over yet'):
            run.command()
        env = {**os.environ, 'DBX_DATALAKE': str(tmp_path), 'DBX_JOURNAL_DATALAKE': str(tmp_path),
               'DBX_JOURNAL_SESSION': run.record['run_session']}
        subprocess.run([sys.executable, '-c', 'import dbx; dbx.pprint()', '--capture-output',
                        "print('from the job') or 7"], env=env, check=True, capture_output=True)
        assert run.command()['session'] == run.record['run_session']
        assert dict(run.command(datalake=str(tmp_path)))['success'] in (True, 'True')
        assert 'from the job' in _printed_(run.command().output)

    def test_the_output_is_read_where_the_journal_is_read_not_where_it_was_written(self, cluster, tmp_path):
        import shutil
        there, here = tmp_path / 'there', tmp_path / 'here'
        run = kube.KubeRun(cluster, 'job-x', dict(run_session='11111111-2222-3333-4444-555555555555',
                                                  datalake=str(there)))
        env = {**os.environ, 'DBX_DATALAKE': str(there), 'DBX_JOURNAL_DATALAKE': str(there),
               'DBX_JOURNAL_SESSION': run.record['run_session']}
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
        first = p.submit('testc').build_tree()
        kubectl.conditions = ['Complete']
        first.wait(poll=0)
        second = p.submit('testc').build_tree()
        runs = dbx.submits.runs(p)
        assert [r.id for r in runs] == [first.id, second.id] and type(runs[0]) is kube.KubeRun
        assert runs[0]._closed_ and not runs[1]._closed_
        assert runs[1].record['manifest'] == kubectl.created[1]
        runs[1].wait(poll=0)
        assert _events_(p.entry) == ['submit:BEGIN', 'submit:END', 'submit:BEGIN', 'submit:END']

    def test_a_pending_job_says_why(self, cluster, kubectl):
        run = kube.submit('true', 'testc')
        kubectl.conditions = ['Pending']
        assert run.status() == 'pending' and run.reason() == 'insufficient quota for cpu'
        df = kube.jobs(['testc'])
        assert list(df['job']) == [run.id] and df['reason'].iloc[0] == 'insufficient quota for cpu'
        assert df['command'].iloc[0] == 'true'

    def test_jobs_are_the_users_unless_all_users(self, cluster, kubectl, monkeypatch):
        kube.submit('true', 'testc')
        cluster.user = 'someone-else'
        assert len(kube.jobs(['testc'])) == 0 and len(kube.jobs(['testc'], all_users=True)) == 1


class TestClusterDryRun:
    def test_it_ships_a_small_run_that_verifies_the_entry_and_records_the_call(self, cluster, kubectl, tmp_path):
        p = Single(spec={'n': 2}, datalake=str(tmp_path), n_gpus=8)
        p.entry.build()
        run = p.submit('testc', dry_run='cluster').build_tree()
        m = kubectl.created[0]
        assert run.id is not None and m['metadata']['name'].startswith('me-single-build-tree-check-')
        assert _env_(m)['DBX_SUBMIT_EXPRESSION'] == f"dbx.submits.verify(dbx.eval({p.quote()!r}).entry, {p.entry.hash!r})"
        assert _env_(m)['DBX_SUBMIT_CALL'].endswith('.entry.build_tree()')
        req = _container_(m)['resources']['requests']
        assert (req['nvidia.com/gpu'], req['cpu']) == ('0', '2') and 'ray start' not in _container_(m)['command'][2]

    def test_verify_reports_the_entry_and_refuses_another_hash(self, tmp_path):
        b = Leaf(datalake=str(tmp_path))
        got = dbx.submits.verify(dbx.eval(b.quote()), b.hash)
        assert got['hash'] == b.hash and got['valid'] is False
        with pytest.raises(ValueError, match='differs'):
            dbx.submits.verify(b, '0' * 8)


# --------------------------------------------------------------------------------------------- #
# kubedash
# --------------------------------------------------------------------------------------------- #

def _queue_(name, quota, used, *, cohort='co', pending=0, admitted=0, resource='nvidia.com/gpu'):
    return {'metadata': {'name': name},
            'spec': {'cohortName': cohort, 'resourceGroups': [{'flavors': [
                {'resources': [{'name': resource, 'nominalQuota': quota}, {'name': 'cpu', 'nominalQuota': '500m'}]}]}]},
            'status': {'flavorsUsage': [{'resources': [{'name': resource, 'total': used, 'borrowed': 0}]}],
                       'pendingWorkloads': pending, 'admittedWorkloads': admitted}}


class TestKubedash:
    @pytest.fixture
    def queues(self, monkeypatch):
        answers = {'a': [_queue_('team-q', '24', '20', admitted=2), _queue_('idle-q', '0', '0'),
                         _queue_('mem-q', '187Gi', '93.5Gi', resource='memory')],
                   'b': [_queue_('team-q', '48', '32', pending=1)]}

        def fake(context, *args, input=None):
            if context not in answers:
                raise RuntimeError(f"Unable to connect to the server: {context}")
            return json.dumps({'items': answers[context]})
        monkeypatch.setattr(kube, '_kubectl_', fake)

    def test_one_row_per_queue_with_quota_or_use(self, queues):
        df = kube.kubedash(['a', 'b'])
        assert list(df['queue']) == ['team-q', 'team-q']
        row = df[df['context'] == 'b'].iloc[0]
        assert (row['quota'], row['used'], row['free'], row['pending']) == (48, 32, 16, 1)

    def test_all_and_a_queue_filter(self, queues):
        assert 'idle-q' in set(kube.kubedash(['a'], all=True)['queue'])
        assert list(kube.kubedash(['a', 'b'], queues='^team')['context']) == ['a', 'b']

    def test_memory_is_in_gib_and_cpu_in_cores(self, queues):
        assert kube.kubedash(['a'], resource='memory', queues='mem')['used'].iloc[0] == 93.5
        assert kube.kubedash(['a'], resource='cpu', queues='team')['quota'].iloc[0] == 0.5

    def test_an_unreachable_context_is_a_row_with_its_error(self, queues):
        df = kube.kubedash(['a', 'gone'])
        assert 'Unable to connect' in df[df['context'] == 'gone']['error'].iloc[0]

    def test_the_console_script_prints_the_table(self, queues, capsys):
        assert kube.kubedash_main(['b', '-q', 'team']) is None
        assert 'team-q' in capsys.readouterr().out
