"""test_datastills.py — Datastill, Datalightning, Dataweights, the cfg_ protocol.

The training-loop tests really do train: a 72-parameter linear model over a
synthetic dataset for two epochs, which is fast and is the only way to pin what
``__build__`` actually leaves behind (checkpoints at both cadences, the
``_COMPLETE`` marker, a resumable latest).
"""
import inspect
import os
from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn

pytest.importorskip('lightning')
import lightning as L

from dbx.datablocks import Datablock
from dbx.datastills import (
    Cfgparam,
    Datalightning,
    Datastill,
    Dataweights,
    cfg_params,
    scaffold_still,
)


# ═══════════════════════════════════════════════════════════════════════
#  Fixtures: a minimal model, module, builder and still
# ═══════════════════════════════════════════════════════════════════════

class ToyModel(nn.Module):
    def __init__(self, *, cfg_width: int = 8, cfg_depth: int = 1,
                 cfg_init_ckpt: str = None):
        super().__init__()
        self.net = nn.Sequential(*[nn.Linear(cfg_width, cfg_width)
                                   for _ in range(cfg_depth)])
        self.init_ckpt = cfg_init_ckpt

    def forward(self, x):
        return self.net(x)


class ToyLightning(L.LightningModule):
    def __init__(self, model, *, cfg_learning_rate: float = 1e-2,
                 cfg_width: int = 8):
        super().__init__()
        self.model, self.lr, self.width = model, cfg_learning_rate, cfg_width

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)

    def training_step(self, batch, i):
        x, _raw = batch
        loss = self.model(x).square().mean()
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, i):
        x, _raw = batch
        self.log('val/loss', self.model(x).square().mean())

    # the data adapter Datastill.dataloaders() asks for
    def val_dataset_kwargs(self):
        return {'return_raw': True}


class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, n, width, return_raw):
        self.n, self.width, self.return_raw = n, width, return_raw

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        x = torch.randn(self.width, generator=torch.Generator().manual_seed(i))
        return (x, i) if self.return_raw else x


class ToyBuilder(Datablock):
    VERSION = 1
    TOPICS = []

    @dataclass
    class VAR(Datablock.VAR):
        n: int = 256
        width: int = 8

    def dataset(self, *, transform=None, return_raw=False):
        return ToyDataset(self.var.n, self.var.width, return_raw)


class ToyStill(Datastill):
    VERSION = 1
    Model, Lightning = ToyModel, ToyLightning

    @dataclass
    class VAR(Datastill.VAR):
        width: int = 8
        depth: int = 1
        init_ckpt: str = None
        learning_rate: float = 1e-2


def make_still(root, **spec):
    builder = ToyBuilder(url=str(root), tag='toydata', spec=dict(n=256, width=8))
    base = dict(
        training_dataset_builder=builder,
        max_epochs=2, batch_size=16, block_shuffle_size=32,
        accumulate_grad_batches=1, precision=None, matmul_precision=None,
        val_every_n_steps=8, val_max_batches=2,
        ckpt_every_n_steps=8, ckpt_every_n_epochs=1,
        train_log_every_n_steps=4, gradient_clip_val=0.0,
    )
    base.update(spec)
    return ToyStill(url=str(root), tag='toy', spec=base, num_workers=0)


# ═══════════════════════════════════════════════════════════════════════
#  The cfg_ protocol
# ═══════════════════════════════════════════════════════════════════════

class TestCfgParams:
    def test_collects_only_prefixed_keyword_arguments(self):
        params = cfg_params({'model': ToyModel, 'lightning': ToyLightning})
        assert set(params) == {'width', 'depth', 'init_ckpt', 'learning_rate'}
        # `model`, the Lightning's first positional, is not a cfg_ argument.
        assert 'model' not in params

    def test_declaration_order_is_preserved_model_first(self):
        params = cfg_params({'model': ToyModel, 'lightning': ToyLightning})
        assert list(params) == ['width', 'depth', 'init_ckpt', 'learning_rate']

    def test_a_name_in_both_roles_is_one_field_feeding_both(self):
        params = cfg_params({'model': ToyModel, 'lightning': ToyLightning})
        assert params['width'].roles == ('model', 'lightning')
        assert params['depth'].roles == ('model',)
        assert params['learning_rate'].roles == ('lightning',)

    def test_disagreeing_defaults_across_roles_raise(self):
        class A:
            def __init__(self, *, cfg_n: int = 1): ...

        class B:
            def __init__(self, *, cfg_n: int = 2): ...

        with pytest.raises(TypeError, match="different defaults"):
            cfg_params({'model': A, 'lightning': B})

    def test_a_positional_only_cfg_argument_raises(self):
        class A:
            def __init__(self, cfg_n, /): ...

        with pytest.raises(TypeError, match="passable by keyword"):
            cfg_params({'model': A})

    def test_a_none_role_is_skipped(self):
        assert cfg_params({'model': ToyModel, 'lightning': None}).keys() == {
            'width', 'depth', 'init_ckpt'}


class TestCfgDispatch:
    def test_var_is_projected_onto_each_constructor(self, tmp_path):
        still = make_still(tmp_path, width=4, depth=2, learning_rate=0.5)
        assert still.model_cfg == {
            'cfg_width': 4, 'cfg_depth': 2, 'cfg_init_ckpt': None}
        assert still.lightning_cfg == {'cfg_learning_rate': 0.5, 'cfg_width': 4}

    def test_the_module_is_built_from_those(self, tmp_path):
        still = make_still(tmp_path, width=4, depth=3, learning_rate=0.5)
        module = still.lightning_module
        assert isinstance(module, ToyLightning)
        assert module.lr == 0.5
        assert len(module.model.net) == 3
        assert module.model.net[0].in_features == 4

    def test_the_module_is_built_once(self, tmp_path):
        still = make_still(tmp_path)
        assert still.lightning_module is still.lightning_module

    def test_a_shared_knob_reaches_both(self, tmp_path):
        still = make_still(tmp_path, width=4)
        assert still.model_cfg['cfg_width'] == still.lightning_cfg['cfg_width'] == 4


class TestCfgDriftIsRefused:
    """The VAR is checked-in source, so it can fall behind what it mirrors."""

    def test_a_missing_var_field_raises_on_construction(self, tmp_path):
        class Drifted(Datastill):
            VERSION = 1
            Model, Lightning = ToyModel, ToyLightning

            @dataclass
            class VAR(Datastill.VAR):
                width: int = 8      # depth / init_ckpt / learning_rate missing

        with pytest.raises(TypeError, match="missing a field"):
            Drifted(url=str(tmp_path), tag='d')

    def test_the_error_names_every_field_to_add(self, tmp_path):
        class Drifted(Datastill):
            VERSION = 1
            Model, Lightning = ToyModel, ToyLightning

            @dataclass
            class VAR(Datastill.VAR):
                width: int = 8

        with pytest.raises(TypeError) as e:
            Drifted(url=str(tmp_path), tag='d')
        for name in ('depth', 'init_ckpt', 'learning_rate'):
            assert name in str(e.value)

    def test_a_cfg_name_colliding_with_a_datastill_var_field_raises(self, tmp_path):
        """`cfg_ckpt` means initial weights; VAR.ckpt means the run to resume."""

        class Colliding(nn.Module):
            def __init__(self, *, cfg_ckpt: str = None):
                super().__init__()

        class CollidingStill(Datastill):
            VERSION = 1
            Model, Lightning = Colliding, ToyLightning

        with pytest.raises(TypeError, match="means something else"):
            CollidingStill(url=str(tmp_path), tag='c')

    def test_an_explicit_mode_still_is_not_checked(self, tmp_path):
        """No Model/Lightning means no cfg surface to drift from."""

        class Explicit(Datastill):
            VERSION = 1

        Explicit(url=str(tmp_path), tag='e')   # must not raise


# ═══════════════════════════════════════════════════════════════════════
#  Identity
# ═══════════════════════════════════════════════════════════════════════

class TestIdentity:
    def test_a_cfg_value_moves_the_hash(self, tmp_path):
        a = make_still(tmp_path, learning_rate=0.1)
        b = make_still(tmp_path, learning_rate=0.2)
        assert a.hash != b.hash

    def test_an_operational_argument_does_not(self, tmp_path):
        builder = ToyBuilder(url=str(tmp_path), tag='toydata', spec=dict(n=256, width=8))
        spec = dict(training_dataset_builder=builder)
        a = ToyStill(url=str(tmp_path), tag='t', spec=spec, num_workers=0, n_devices=1)
        b = ToyStill(url=str(tmp_path), tag='t', spec=spec, num_workers=8, n_devices=4,
                     prefetch_factor=16, debug_share_train_val=True)
        assert a.hash == b.hash

    @pytest.mark.pinned
    def test_datastill_takes_the_corrected_norm(self):
        """A new class must NOT inherit the legacy rendering."""
        assert Datastill.LEGACY_NORM is False
        assert Datalightning.LEGACY_NORM is False
        assert Dataweights.LEGACY_NORM is False

    @pytest.mark.pinned
    def test_datalightning_declares_no_topics(self):
        """`TOPICS = []` and no TOPICS render differently into the identity.

        signature_topics() answers ("topics:None",) for a class with no TOPICS
        and () for one declaring an empty list, so declaring the empty list on
        the base would move the hash of every subclass that already has
        artifacts -- which is exactly what IJEPAsaurUSLightning is.
        """
        assert 'TOPICS' not in vars(Datalightning)

    @pytest.mark.pinned
    def test_datastill_var_field_order(self):
        """Load-bearing for a LEGACY_NORM subclass: __expand_spec__ renders a
        legacy block's spec in __dataclass_fields__ order. Reordering, or
        inserting in the middle, re-keys every such block.
        """
        assert list(Datastill.VAR.__dataclass_fields__) == [
            'lightning', 'training_dataset_builder', 'validation_dataset_builder',
            'train_val_split', 'dataset_seed', 'block_shuffle_size', 'ckpt',
            'from_scratch', 'reset_optimizer_state', 'max_epochs',
            'max_training_steps', 'train_log_every_n_steps', 'val_every_n_steps',
            'val_max_batches', 'val_shuffle', 'val_seed', 'gradient_clip_val',
            'gradient_clip_algorithm', 'ckpt_every_n_steps', 'ckpt_every_n_epochs',
            'precision', 'matmul_precision', 'accumulate_grad_batches',
            'batch_size', 'still_seed',
        ]


# ═══════════════════════════════════════════════════════════════════════
#  Dataloaders
# ═══════════════════════════════════════════════════════════════════════

class TestDataloaders:
    def test_the_split_is_disjoint_and_exhaustive(self, tmp_path):
        train, val = make_still(tmp_path, train_val_split=0.75).dataloaders()
        assert len(train.dataset) + len(val.dataset) == 256
        assert not set(train.dataset.indices) & set(val.dataset.indices)

    def test_the_split_is_block_granular(self, tmp_path):
        """Every group is a union of contiguous runs, or shard locality is lost."""
        train, _ = make_still(tmp_path, block_shuffle_size=32).dataloaders()
        idx = sorted(train.dataset.indices)
        runs, start = [], idx[0]
        for a, b in zip(idx, idx[1:]):
            if b != a + 1:
                runs.append((start, a)); start = b
        runs.append((start, idx[-1]))
        for lo, hi in runs:
            assert (hi - lo + 1) % 32 == 0, f"run {(lo, hi)} is not whole blocks"

    def test_the_split_is_seeded(self, tmp_path):
        a, _ = make_still(tmp_path, dataset_seed=1).dataloaders()
        b, _ = make_still(tmp_path, dataset_seed=1).dataloaders()
        c, _ = make_still(tmp_path, dataset_seed=2).dataloaders()
        assert a.dataset.indices == b.dataset.indices
        assert a.dataset.indices != c.dataset.indices

    def test_debug_share_train_val_defeats_the_split(self, tmp_path):
        train, val = make_still(tmp_path).dataloaders(debug_share_train_val=True)
        assert train.dataset is val.dataset

    def test_a_separate_validation_builder_is_used_as_is(self, tmp_path):
        val_builder = ToyBuilder(url=str(tmp_path), tag='valdata',
                                 spec=dict(n=64, width=8))
        train, val = make_still(
            tmp_path, validation_dataset_builder=val_builder).dataloaders()
        # No Subset: two independently-built datasets, at their full lengths.
        assert len(train.dataset) == 256
        assert len(val.dataset) == 64

    def test_val_shuffle_off_means_no_val_sampler(self, tmp_path):
        _, val = make_still(tmp_path, val_shuffle=False).dataloaders()
        assert not isinstance(val.sampler, Datastill.Sampler)

    def test_val_shuffle_on_gives_a_fixed_order_sampler(self, tmp_path):
        _, val = make_still(tmp_path, val_shuffle=True).dataloaders()
        assert isinstance(val.sampler, Datastill.Sampler)
        before = list(iter(val.sampler))
        val.sampler.set_epoch(7)
        assert list(iter(val.sampler)) == before

    def test_the_train_sampler_reshuffles_per_epoch(self, tmp_path):
        train, _ = make_still(tmp_path).dataloaders()
        before = list(iter(train.sampler))
        train.sampler.set_epoch(1)
        assert list(iter(train.sampler)) != before

    def test_the_val_loader_is_sized_from_val_max_batches(self, tmp_path):
        still = ToyStill(
            url=str(tmp_path), tag='toy', num_workers=12, prefetch_factor=4,
            spec=dict(
                training_dataset_builder=ToyBuilder(
                    url=str(tmp_path), tag='d', spec=dict(n=256, width=8)),
                val_max_batches=1,
            ),
        )
        train, val = still.dataloaders()
        assert train.num_workers == 12
        assert val.num_workers == 1
        assert val.num_workers * val.prefetch_factor <= 2

    def test_no_training_builder_raises_rather_than_failing_later(self, tmp_path):
        still = ToyStill(url=str(tmp_path), tag='toy', spec=dict())
        with pytest.raises(ValueError, match="nothing to train on"):
            still.dataloaders()


class TestLoaderHardening:
    """Every loader a still builds must carry its ``worker_init_fn``.

    A loader that omits it has no symptom until a forked worker deadlocks on a
    lock copied while held -- possibly hours in. Previously asserted by
    counting ``DataLoader(`` against ``worker_init_fn=`` in the source of each
    Still's ``dataloaders``; now structural, since every loader is constructed
    in one place.
    """

    def test_every_loader_is_built_through_the_one_constructor(self):
        src = inspect.getsource(Datastill.dataloaders)
        assert 'DataLoader(' not in src, (
            "dataloaders() constructs a loader directly; route it through "
            "_dataloader() so worker_init_fn cannot be forgotten on one of them"
        )
        assert src.count('self._dataloader(') == 2

    def test_the_one_constructor_always_passes_worker_init_fn(self):
        src = inspect.getsource(Datastill._dataloader)
        assert 'worker_init_fn=self.worker_init_fn()' in src

    def test_a_subclass_hook_reaches_both_loaders(self, tmp_path):
        sentinel = _noop_worker_init

        class HardenedStill(ToyStill):
            VERSION = 1

            def worker_init_fn(self):
                return sentinel

        builder = ToyBuilder(url=str(tmp_path), tag='d', spec=dict(n=256, width=8))
        still = HardenedStill(url=str(tmp_path), tag='h', num_workers=2,
                              spec=dict(training_dataset_builder=builder))
        train, val = still.dataloaders()
        assert train.worker_init_fn is sentinel
        assert val.worker_init_fn is sentinel


def _noop_worker_init(worker_id):
    """A module-level function, so it is picklable as a spawn-context loader needs."""


# ═══════════════════════════════════════════════════════════════════════
#  Build
# ═══════════════════════════════════════════════════════════════════════

class TestBuild:
    @pytest.fixture(scope='class')
    def trained(self, tmp_path_factory):
        root = tmp_path_factory.mktemp('trained')
        still = make_still(root)
        assert still.valid() is False
        still.build()
        return still

    def test_valid_only_after_the_complete_marker(self, trained):
        assert trained.valid() is True
        assert os.path.exists(os.path.join(trained._local_ckpts_dir, '_COMPLETE'))

    def test_checkpoints_at_both_cadences(self, trained):
        names = [n for n in os.listdir(trained.dirpath('ckpts')) if n.endswith('.ckpt')]
        assert names
        assert {trained._ckpt_step(n) for n in names} >= {8, 16}

    def test_find_latest_ckpt_is_the_highest_step(self, trained):
        latest = os.path.basename(str(trained.find_latest_ckpt()))
        steps = [trained._ckpt_step(n)
                 for n in os.listdir(trained.dirpath('ckpts')) if n.endswith('.ckpt')]
        assert trained._ckpt_step(latest) == max(steps)

    def test_the_latest_is_loadable(self, trained):
        path = trained.find_latest_ckpt(pull=True)
        assert path is not None
        state = torch.load(path, weights_only=False)
        assert 'state_dict' in state

    def test_tensorboard_logs_were_written(self, trained):
        runs = os.listdir(trained.dirpath('logs'))
        assert any(r.startswith('run_') for r in runs), runs

    def test_from_scratch_ignores_its_own_latest(self, trained):
        fresh = trained.set(tag='toy').replace(spec={**trained.spec, 'from_scratch': True})
        assert fresh.var.from_scratch is True
        # _resume_plan consults var.ckpt only; with neither, nothing to resume.
        assert fresh._resume_plan() is None

    def test_unsafe_clear_removes_the_local_marker_too(self, tmp_path):
        still = make_still(tmp_path)
        still.build()
        assert still.valid() is True
        still.UNSAFE_clear(OVERRIDE=True)
        assert still.valid() is False


class TestWarmStart:
    """``VAR.ckpt`` holding another *block*: its weights, not its run.

    The failure this prevents is quiet. A full Lightning resume restores the
    epoch counter too, so a source that already reached its own ``max_epochs``
    leaves the new run with nothing to do: ``fit()`` returns at once, the
    ``_COMPLETE`` marker is written, and an untrained model sits at a key that
    claims to be trained.
    """

    @pytest.fixture(scope='class')
    def source(self, tmp_path_factory):
        """A finished run to start from -- 2 epochs, i.e. its max_epochs."""
        still = make_still(tmp_path_factory.mktemp('source'))
        still.build()
        return still

    def test_a_block_is_a_warm_start(self, source, tmp_path):
        still = make_still(tmp_path, ckpt=source)
        ckpt, own_it, warm_start = still._resolve_resume_ckpt()
        assert warm_start is True
        assert own_it is False
        assert ckpt == source.find_latest_ckpt(pull=True)

    def test_a_path_is_a_full_resume(self, source, tmp_path):
        """The escape hatch: pointing at the file, not the block, is how you
        ask for the optimizer state back."""
        path = source.find_latest_ckpt(pull=True)
        still = make_still(tmp_path, ckpt=path)
        ckpt, own_it, warm_start = still._resolve_resume_ckpt()
        assert warm_start is False
        assert ckpt == path

    def test_this_runs_own_checkpoint_wins(self, source, tmp_path):
        """Once it has trained, a restart continues *this* run -- the warm-start
        source is history, not something to fall back to on every launch."""
        still = make_still(tmp_path, ckpt=source)
        still.build()
        ckpt, own_it, warm_start = still._resolve_resume_ckpt()
        assert warm_start is False
        assert own_it is True

    def test_a_source_with_no_checkpoints_raises(self, tmp_path):
        untrained = make_still(tmp_path / 'untrained')
        still = make_still(tmp_path / 'still', ckpt=untrained)
        with pytest.raises(FileNotFoundError, match='no checkpoints'):
            still._resolve_resume_ckpt()

    def test_the_banner_says_weights_only(self, source, tmp_path):
        plan = make_still(tmp_path, ckpt=source)._resume_plan()
        assert 'warm start, weights only' in plan
        assert source.key in plan

    def test_a_full_resume_plan_is_the_checkpoint_itself(self, source, tmp_path):
        path = source.find_latest_ckpt(pull=True)
        assert make_still(tmp_path, ckpt=path)._resume_plan() == path

    def test_the_divergence_report_is_skipped_for_a_warm_start(self, source, tmp_path, caplog):
        """The plan names the block, the checkpoint is a file -- comparing the
        two would cry wolf on every warm start."""
        still = make_still(tmp_path, ckpt=source)
        still._report_resume_divergence(
            source.find_latest_ckpt(pull=True), still._resume_plan(), warm_start=True,
        )
        assert 'NOT the one in the banner' not in caplog.text

    def test_it_trains_its_own_schedule_from_step_zero(self, source, tmp_path):
        """The behaviour the rest of this class is about, end to end: the
        warm-started run does its own 2 epochs rather than inheriting the
        source's exhausted epoch counter and stopping instantly."""
        still = make_still(tmp_path, ckpt=source)
        still.build()
        assert still.valid() is True
        steps = {still._ckpt_step(n)
                 for n in os.listdir(still.dirpath('ckpts')) if n.endswith('.ckpt')}
        source_steps = {source._ckpt_step(n)
                        for n in os.listdir(source.dirpath('ckpts')) if n.endswith('.ckpt')}
        assert steps == source_steps, "should have run its own schedule, from 0"

    def test_the_weights_are_the_sources(self, source, tmp_path):
        """...and they are the source's weights, not a fresh init."""
        still = make_still(tmp_path, ckpt=source)
        model = still.lightning_module
        key = next(k for k in model.state_dict() if k.endswith('.weight'))
        before = model.state_dict()[key].clone()
        still._load_weights_only(model, source.find_latest_ckpt(pull=True), why='test')
        after = model.state_dict()[key]
        expected = torch.load(
            source.find_latest_ckpt(pull=True), weights_only=False,
        )['state_dict'][key]
        assert torch.equal(after, expected)
        assert not torch.equal(after, before)


class TestLoadWeightsOnly:
    """The load itself -- what ``strict=False`` will and will not catch."""

    @pytest.fixture
    def still(self, tmp_path):
        return make_still(tmp_path)

    def _ckpt(self, tmp_path, payload):
        path = tmp_path / 'hand.ckpt'
        torch.save(payload, path)
        return str(path)

    def test_a_checkpoint_without_a_state_dict_says_so(self, still, tmp_path):
        path = self._ckpt(tmp_path, {'epoch': 3})
        with pytest.raises(KeyError, match='state_dict'):
            still._load_weights_only(still.lightning_module, path, why='test')

    def test_a_checkpoint_of_another_architecture_is_refused(self, still, tmp_path):
        """strict=False silently ignores names it doesn't know, so a warm start
        pointed at the wrong model loads *nothing* and trains from random init
        while the banner claims otherwise."""
        path = self._ckpt(tmp_path, {'state_dict': {'nothing.like.it': torch.zeros(3)}})
        with pytest.raises(ValueError, match='nothing loaded'):
            still._load_weights_only(still.lightning_module, path, why='test')

    def test_a_partial_match_is_allowed(self, still, tmp_path):
        """A changed head is the reason this loads with strict=False at all."""
        model = still.lightning_module
        state = dict(model.state_dict())
        state['some.extra.head'] = torch.zeros(3)
        path = self._ckpt(tmp_path, {'state_dict': state})
        still._load_weights_only(model, path, why='test')   # does not raise


class TestCkptStep:
    @pytest.mark.parametrize('name,step', [
        ('epoch=003-step=0001234.ckpt', 1234),
        ('epoch=000-step=step=0000008.ckpt', 8),   # Lightning's doubled form
        ('no-step-here.ckpt', -1),
    ])
    def test_parses_the_step_out_of_the_filename(self, name, step):
        assert Datastill._ckpt_step(name) == step

    def test_sorts_numerically_not_lexically(self):
        names = ['step=9.ckpt', 'step=10.ckpt', 'step=100.ckpt']
        assert sorted(names, key=Datastill._ckpt_step) == names


# ═══════════════════════════════════════════════════════════════════════
#  Datalightning
# ═══════════════════════════════════════════════════════════════════════

class TestDatalightning:
    def test_valid_reflects_nested_var_blocks(self, tmp_path):
        class Upstream(Datablock):
            VERSION = 1
            TOPICS = {'thing': 'thing.txt'}

            def __build__(self):
                with self.fs.open(self.path('thing', ensure_dirpath=True), 'w') as f:
                    f.write('x')
                return self

        class Wrapper(Datalightning):
            VERSION = 1

            @dataclass
            class VAR(Datalightning.VAR):
                upstream: object = None

            def __lightning_module__(self):
                return ToyLightning(ToyModel())

        up = Upstream(url=str(tmp_path), tag='u')
        wrapper = Wrapper(url=str(tmp_path), tag='w', spec=dict(upstream=up))
        # The whole point: a topicless block must NOT report itself valid while
        # an upstream it needs is unbuilt, or build_tree() skips the subtree.
        assert wrapper.valid() is False
        up.build()
        assert wrapper.valid() is True

    def test_build_skips_rather_than_reaching_build(self, tmp_path):
        """validate_vars=False does NOT get you to __build__.

        valid_var() answers True unconditionally when validation is off, so
        valid() is True and build() skips the block as already done. Worth
        pinning because the raise below is documented as the guard for exactly
        this route, and it is not: the skip is.
        """
        class Upstream(Datablock):
            VERSION = 1
            TOPICS = {'thing': 'thing.txt'}

        class Wrapper(Datalightning):
            VERSION = 1

            @dataclass
            class VAR(Datalightning.VAR):
                upstream: object = None

            def __lightning_module__(self):
                return ToyLightning(ToyModel())

        up = Upstream(url=str(tmp_path), tag='u')
        assert up.valid() is False
        w = Wrapper(url=str(tmp_path), tag='w', validate_vars=False,
                    spec=dict(upstream=up))
        assert w.valid() is True
        w.build()      # skipped, not raised

    def test_calling_build_directly_raises(self, tmp_path):
        """The guard that does fire: nothing may quietly no-op here."""
        class Wrapper(Datalightning):
            VERSION = 1

            def __lightning_module__(self):
                return ToyLightning(ToyModel())

        with pytest.raises(NotImplementedError, match="unreachable"):
            Wrapper(url=str(tmp_path), tag='w').__build__()

    def test_the_module_is_cached(self, tmp_path):
        class Wrapper(Datalightning):
            VERSION = 1

            def __lightning_module__(self):
                return ToyLightning(ToyModel())

        w = Wrapper(url=str(tmp_path), tag='w')
        assert w.lightning_module is w.lightning_module

    def test_an_unimplemented_module_says_so(self, tmp_path):
        class Wrapper(Datalightning):
            VERSION = 1

        with pytest.raises(NotImplementedError, match="__lightning_module__"):
            Wrapper(url=str(tmp_path), tag='w').lightning_module

    def test_a_datalightning_in_var_supplies_the_still_its_module(self, tmp_path):
        class Wrapper(Datalightning):
            VERSION = 1

            def __lightning_module__(self):
                return ToyLightning(ToyModel(), cfg_learning_rate=0.123)

        class ExplicitStill(Datastill):
            VERSION = 1

        wrapper = Wrapper(url=str(tmp_path), tag='w')
        still = ExplicitStill(url=str(tmp_path), tag='e', spec=dict(lightning=wrapper))
        assert still.lightning_module.lr == 0.123

    def test_neither_mode_configured_says_so(self, tmp_path):
        class Empty(Datastill):
            VERSION = 1

        with pytest.raises(ValueError, match="nothing to train"):
            Empty(url=str(tmp_path), tag='x').lightning_module


# ═══════════════════════════════════════════════════════════════════════
#  Dataweights
# ═══════════════════════════════════════════════════════════════════════

class TestDataweights:
    def test_no_ckpt_is_trivially_valid_and_has_no_local_path(self, tmp_path):
        w = Dataweights(url=str(tmp_path), tag='none', spec=dict(ckpt=None))
        assert w.valid() is True
        w.build()
        assert w.path_local() is None

    def test_fetches_persists_and_hands_back_a_local_path(self, tmp_path):
        src = tmp_path / 'src.pt'
        torch.save({'k': torch.zeros(3)}, src)
        w = Dataweights(url=str(tmp_path / 'store'), tag='w',
                        spec=dict(ckpt=str(src)))
        assert w.valid() is False
        w.build()
        assert w.valid() is True
        local = w.path_local()
        assert os.path.isfile(local)
        assert 'k' in torch.load(local, weights_only=False)

    def test_a_truncated_blob_is_not_valid(self, tmp_path):
        src = tmp_path / 'src.pt'
        torch.save({'k': torch.zeros(3)}, src)
        w = Dataweights(url=str(tmp_path / 'store'), tag='w', spec=dict(ckpt=str(src)))
        w.build()
        with open(w.path('weights'), 'r+b') as f:
            f.truncate(16)
        assert w.valid() is False

    def test_source_url_is_the_extension_point(self, tmp_path):
        src = tmp_path / 'real.pt'
        torch.save({'k': 1}, src)

        class Registry(Dataweights):
            VERSION = 1
            WEIGHTS = {'published-v1': str(src)}

            def source_url(self):
                try:
                    return self.WEIGHTS[self.var.ckpt]
                except KeyError:
                    raise ValueError(f"unknown checkpoint {self.var.ckpt!r}") from None

        ok = Registry(url=str(tmp_path / 's1'), tag='a', spec=dict(ckpt='published-v1'))
        ok.build()
        assert ok.valid() is True

        bad = Registry(url=str(tmp_path / 's2'), tag='b', spec=dict(ckpt='nope'))
        with pytest.raises(ValueError, match="unknown checkpoint"):
            bad.build()


# ═══════════════════════════════════════════════════════════════════════
#  scaffold_still
# ═══════════════════════════════════════════════════════════════════════

class TestScaffold:
    def _exec(self, src):
        ns = {'Datastill': Datastill, 'dataclass': dataclass,
              'ToyModel': ToyModel, 'ToyLightning': ToyLightning}
        exec(compile(src, '<scaffold>', 'exec'), ns)
        return ns

    def test_the_generated_var_covers_the_whole_cfg_surface(self, tmp_path):
        src = scaffold_still(ToyModel, ToyLightning, name='Gen')
        cls = self._exec(src)['Gen']
        declared = set(cls.VAR.__dataclass_fields__)
        assert {'width', 'depth', 'init_ckpt', 'learning_rate'} <= declared

    def test_the_generated_class_constructs_and_dispatches(self, tmp_path):
        src = scaffold_still(ToyModel, ToyLightning, name='Gen')
        cls = self._exec(src)['Gen']
        still = cls(url=str(tmp_path), tag='g', spec=dict(learning_rate=0.25))
        assert still.lightning_module.lr == 0.25

    def test_the_generated_class_passes_its_own_drift_check(self, tmp_path):
        """Which is the point: generation and the check share one source of truth."""
        src = scaffold_still(ToyModel, ToyLightning, name='Gen')
        cls = self._exec(src)['Gen']
        cls(url=str(tmp_path), tag='g').__check_cfg__()

    def test_generation_is_deterministic(self):
        a = scaffold_still(ToyModel, ToyLightning, name='Gen')
        b = scaffold_still(ToyModel, ToyLightning, name='Gen')
        assert a == b

    def test_the_entrypoint_exposes_var_prefixed_arguments(self, tmp_path):
        src = scaffold_still(ToyModel, ToyLightning, name='Gen', entrypoint='gen_still')
        ns = self._exec(src)
        still = ns['gen_still'](url=str(tmp_path), tag='g', var_learning_rate=0.75)
        assert still.tag == 'g'
        assert still.var.learning_rate == 0.75

    def test_a_mutable_default_becomes_a_default_factory(self):
        class Mutable(nn.Module):
            def __init__(self, *, cfg_layers: list = [1, 2]):
                super().__init__()

        src = scaffold_still(Mutable, ToyLightning, name='Gen')
        assert 'field(default_factory=lambda: [1, 2])' in src

    def test_a_colliding_name_is_refused_at_generation_time(self):
        class Colliding(nn.Module):
            def __init__(self, *, cfg_batch_size: int = 4):
                super().__init__()

        with pytest.raises(ValueError, match="different meaning"):
            scaffold_still(Colliding, ToyLightning, name='Gen')

    def test_no_cfg_surface_at_all_says_so(self):
        class Plain(nn.Module):
            def __init__(self):
                super().__init__()

        class PlainLightning(L.LightningModule):
            def __init__(self, model):
                super().__init__()

        with pytest.raises(ValueError, match="nothing to mirror"):
            scaffold_still(Plain, PlainLightning, name='Gen')


# ═══════════════════════════════════════════════════════════════════════
#  export
# ═══════════════════════════════════════════════════════════════════════

class TestExport:
    def test_bakes_the_configuration_as_literals(self, tmp_path):
        src = make_still(tmp_path, width=4, learning_rate=0.5).export(ckpt='/w.ckpt')
        assert "'cfg_width': 4" in src
        assert "'cfg_learning_rate': 0.5" in src
        assert "CKPT = '/w.ckpt'" in src

    def test_the_generated_module_imports_no_dbx(self, tmp_path):
        src = make_still(tmp_path).export(ckpt=None)
        imports = [l for l in src.splitlines()
                   if l.startswith('import ') or l.startswith('from ')]
        assert imports
        assert not [l for l in imports if 'dbx' in l], imports

    def test_one_import_line_per_module(self, tmp_path):
        src = make_still(tmp_path).export(ckpt=None)
        assert src.count('import ToyModel, ToyLightning') == 1

    def test_the_generated_module_runs(self, tmp_path):
        path = tmp_path / 'gen.py'
        make_still(tmp_path, width=4).export(path=str(path), ckpt=None)
        ns = {'__name__': 'gen'}
        exec(compile(path.read_text(), str(path), 'exec'), ns)
        module = ns['build_model'](None)
        assert module.model.net[0].in_features == 4

    def test_refuses_to_overwrite_a_file_that_is_not_ours(self, tmp_path):
        path = tmp_path / 'mine.py'
        path.write_text("# hand-written, do not clobber\n")
        with pytest.raises(FileExistsError, match="not generated by export"):
            make_still(tmp_path).export(path=str(path), ckpt=None)
        assert 'hand-written' in path.read_text()

    def test_regenerates_over_its_own_output(self, tmp_path):
        path = tmp_path / 'gen.py'
        make_still(tmp_path, width=4).export(path=str(path), ckpt=None)
        make_still(tmp_path, width=6).export(path=str(path), ckpt=None)
        assert "'cfg_width': 6" in path.read_text()

    def test_overwrite_false_refuses_even_its_own_output(self, tmp_path):
        path = tmp_path / 'gen.py'
        make_still(tmp_path).export(path=str(path), ckpt=None)
        with pytest.raises(FileExistsError, match="overwrite=True"):
            make_still(tmp_path).export(path=str(path), ckpt=None, overwrite=False)

    def test_explicit_mode_has_no_generic_export(self, tmp_path):
        class Explicit(Datastill):
            VERSION = 1

        with pytest.raises(NotImplementedError, match="cfg mode"):
            Explicit(url=str(tmp_path), tag='e').export()
