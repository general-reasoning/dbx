"""test_stills.py — Still, the builder blocks it trains from, Weights.

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

# NOT importorskip: lightning is a declared dependency of this package
# (the `lightning` extra, which `all` pulls in), so its absence is a broken
# environment rather than an optional feature. Skipping here is how this
# whole module went unrun against a stale env.
import lightning as L

from dbx.datablocks import Datablock
from dbx.stills import (
    CheckpointBuilder,
    DatasetBuilder,
    LightningBuilder,
    ModelBuilder,
    Still,
    Weights,
)


# ═══════════════════════════════════════════════════════════════════════
#  Fixtures: a minimal model, module, and one builder block per collaborator
# ═══════════════════════════════════════════════════════════════════════

class ToyModel(nn.Module):
    def __init__(self, *, width: int = 8, depth: int = 1, init_ckpt: str = None):
        super().__init__()
        self.net = nn.Sequential(*[nn.Linear(width, width) for _ in range(depth)])
        self.init_ckpt = init_ckpt

    def forward(self, x):
        return self.net(x)


class ToyLightning(L.LightningModule):
    def __init__(self, model, *, learning_rate: float = 1e-2, width: int = 8):
        super().__init__()
        self.model, self.lr, self.width = model, learning_rate, width

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

    # the data adapter Still.dataloaders() asks for
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


class ToyModelBuilder(ModelBuilder):
    VERSION = 1

    @dataclass
    class VAR(Datablock.VAR):
        width: int = 8
        depth: int = 1
        init_ckpt: str = None

    def model(self):
        return ToyModel(width=self.var.width, depth=self.var.depth,
                        init_ckpt=self.var.init_ckpt)


class ToyLightningBuilder(LightningBuilder):
    VERSION = 1

    @dataclass
    class VAR(LightningBuilder.VAR):
        model_builder: ModelBuilder | str | None = None
        learning_rate: float = 1e-2
        width: int = 8

    def __lightning_module__(self):
        return ToyLightning(self.var.model_builder.model(),
                            learning_rate=self.var.learning_rate,
                            width=self.var.width)


class ToyBuilder(DatasetBuilder):
    VERSION = 1
    TOPICS = []

    @dataclass
    class VAR(Datablock.VAR):
        n: int = 256
        width: int = 8

    def dataset(self, *, transform=None, return_raw=False):
        return ToyDataset(self.var.n, self.var.width, return_raw)


class ToyStill(Still):
    VERSION = 1


#: What ToyModelBuilder / ToyLightningBuilder take, as opposed to what the
#: still itself takes. make_still() routes each kwarg to whichever owns it.
MODEL_KNOBS = ('width', 'depth', 'init_ckpt')
LIGHTNING_KNOBS = ('learning_rate',)


def toy_builders(root, **knobs):
    """The four required builder blocks, with *knobs* routed to their owners."""
    model = ToyModelBuilder(
        url=str(root), tag='toymodel',
        spec={k: v for k, v in knobs.items() if k in MODEL_KNOBS})
    lightning = ToyLightningBuilder(
        url=str(root), tag='toylightning',
        spec=dict(model_builder=model, width=knobs.get('width', 8),
                  **{k: v for k, v in knobs.items() if k in LIGHTNING_KNOBS}))
    data = ToyBuilder(url=str(root), tag='toydata',
                      spec=dict(n=256, width=knobs.get('width', 8)))
    return dict(model_builder=model, lightning_builder=lightning,
                training_dataset_builder=data, validation_dataset_builder=data)


def make_still(root, **spec):
    knobs = {k: spec.pop(k) for k in (*MODEL_KNOBS, *LIGHTNING_KNOBS) if k in spec}
    base = dict(
        **toy_builders(root, **knobs),
        max_epochs=2, batch_size=16, block_shuffle_size=32,
        accumulate_grad_batches=1, precision=None, matmul_precision=None,
        val_every_n_steps=8, val_max_batches=2,
        ckpt_every_n_steps=8, ckpt_every_n_epochs=1,
        train_log_every_n_steps=4, gradient_clip_val=0.0,
    )
    base.update(spec)
    return ToyStill(url=str(root), tag='toy', spec=base, num_workers=0)


# ═══════════════════════════════════════════════════════════════════════
#  Identity
# ═══════════════════════════════════════════════════════════════════════

class TestIdentity:
    def test_a_builders_configuration_moves_the_hash(self, tmp_path):
        a = make_still(tmp_path, learning_rate=0.1)
        b = make_still(tmp_path, learning_rate=0.2)
        assert a.hash != b.hash

    def test_a_different_builder_moves_the_hash(self, tmp_path):
        assert make_still(tmp_path, width=4).hash != make_still(tmp_path, width=8).hash

    def test_an_operational_argument_does_not(self, tmp_path):
        spec = toy_builders(tmp_path)
        a = ToyStill(url=str(tmp_path), tag='t', spec=spec, num_workers=0, n_devices=1)
        b = ToyStill(url=str(tmp_path), tag='t', spec=spec, num_workers=8, n_devices=4,
                     prefetch_factor=16, debug_share_train_val=True)
        assert a.hash == b.hash

    @pytest.mark.pinned
    def test_datastill_takes_the_corrected_norm(self):
        """A new class must NOT inherit the legacy rendering."""
        assert Still.LEGACY_NORM is False
        assert LightningBuilder.LEGACY_NORM is False
        assert Weights.LEGACY_NORM is False

    @pytest.mark.pinned
    def test_datalightning_declares_no_topics(self):
        """`TOPICS = []` and no TOPICS render differently into the identity.

        signature_topics() answers ("topics:None",) for a class with no TOPICS
        and () for one declaring an empty list, so declaring the empty list on
        the base would move the hash of every subclass that already has
        artifacts -- which is exactly what IJEPAsaurUSLightning is.
        """
        assert 'TOPICS' not in vars(LightningBuilder)

    @pytest.mark.pinned
    def test_datastill_var_field_order(self):
        """Load-bearing for a LEGACY_NORM subclass: __expand_spec__ renders a
        legacy block's spec in __dataclass_fields__ order. Reordering, or
        inserting in the middle, re-keys every such block.

        RE-PINNED 2026-09-16, deliberately and once: builder blocks replaced
        the cfg_ surface, `lightning` became `lightning_builder`, `ckpt` became
        `ckpt_builder`, `model_builder` arrived, and all five moved to the
        front. That re-keyed every existing still, which is why Still.VERSION
        went to 2 in the same change. Reordering from HERE re-keys again, and
        this pin is the thing that says so.
        """
        assert list(Still.VAR.__dataclass_fields__) == [
            'model_builder', 'lightning_builder', 'training_dataset_builder',
            'validation_dataset_builder', 'ckpt_builder',
            'train_val_split', 'dataset_seed', 'block_shuffle_size',
            'from_scratch', 'reset_optimizer_state', 'max_epochs',
            'max_training_steps', 'train_log_every_n_steps', 'val_every_n_steps',
            'val_max_batches', 'val_shuffle', 'val_seed', 'gradient_clip_val',
            'gradient_clip_algorithm', 'ckpt_every_n_steps', 'ckpt_every_n_epochs',
            'precision', 'matmul_precision', 'accumulate_grad_batches',
            'batch_size', 'still_seed',
        ]

    @pytest.mark.pinned
    def test_still_version_records_the_builder_change(self):
        """VERSION 2 is what distinguishes a builder still from a cfg_ one."""
        assert Still.VERSION == 2


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
        assert not isinstance(val.sampler, Still.Sampler)

    def test_val_shuffle_on_gives_a_fixed_order_sampler(self, tmp_path):
        _, val = make_still(tmp_path, val_shuffle=True).dataloaders()
        assert isinstance(val.sampler, Still.Sampler)
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
            spec=dict(**toy_builders(tmp_path), val_max_batches=1),
        )
        train, val = still.dataloaders()
        assert train.num_workers == 12
        assert val.num_workers == 1
        assert val.num_workers * val.prefetch_factor <= 2

    def test_a_missing_builder_raises_on_construction_not_hours_later(self, tmp_path):
        with pytest.raises(TypeError, match="model_builder"):
            ToyStill(url=str(tmp_path), tag='toy', spec=dict())

    def test_the_error_names_every_builder_that_is_missing(self, tmp_path):
        spec = toy_builders(tmp_path)
        del spec['training_dataset_builder'], spec['validation_dataset_builder']
        with pytest.raises(TypeError) as e:
            ToyStill(url=str(tmp_path), tag='toy', spec=spec)
        assert 'training_dataset_builder' in str(e.value)
        assert 'validation_dataset_builder' in str(e.value)

    def test_ckpt_builder_is_not_required(self, tmp_path):
        assert ToyStill(url=str(tmp_path), tag='toy',
                        spec=toy_builders(tmp_path)).var.ckpt_builder is None


class TestLoaderHardening:
    """Every loader a still builds must carry its ``dataloader_worker_init_fn``.

    A loader that omits it has no symptom until a forked worker deadlocks on a
    lock copied while held -- possibly hours in. Previously asserted by
    counting ``DataLoader(`` against ``worker_init_fn=`` in the source of each
    Still's ``dataloaders``; now structural, since every loader is constructed
    in one place.
    """

    def test_every_loader_is_built_through_the_one_constructor(self):
        src = inspect.getsource(Still.dataloaders)
        assert 'DataLoader(' not in src, (
            "dataloaders() constructs a loader directly; route it through "
            "_dataloader_() so dataloader_worker_init_fn cannot be forgotten on one of them"
        )
        assert src.count('self._dataloader_(') == 2

    def test_the_one_constructor_always_passes_the_worker_init_fn(self):
        src = inspect.getsource(Still._dataloader_)
        assert 'worker_init_fn=self.dataloader_worker_init_fn()' in src

    def test_a_subclass_hook_reaches_both_loaders(self, tmp_path):
        sentinel = _noop_dataloader_worker_init_

        class HardenedStill(ToyStill):
            VERSION = 1

            def dataloader_worker_init_fn(self):
                return sentinel

        still = HardenedStill(url=str(tmp_path), tag='h', num_workers=2,
                              spec=toy_builders(tmp_path))
        train, val = still.dataloaders()
        assert train.worker_init_fn is sentinel
        assert val.worker_init_fn is sentinel


def _noop_dataloader_worker_init_(worker_id):
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
        assert os.path.exists(os.path.join(trained._local_ckpts_dir_, '_COMPLETE'))

    def test_checkpoints_at_both_cadences(self, trained):
        names = [n for n in os.listdir(trained.dirpath('ckpts')) if n.endswith('.ckpt')]
        assert names
        assert {trained._ckpt_step_(n) for n in names} >= {8, 16}

    def test_find_latest_ckpt_is_the_highest_step(self, trained):
        latest = os.path.basename(str(trained.find_latest_ckpt()))
        steps = [trained._ckpt_step_(n)
                 for n in os.listdir(trained.dirpath('ckpts')) if n.endswith('.ckpt')]
        assert trained._ckpt_step_(latest) == max(steps)

    def test_the_latest_is_loadable(self, trained):
        path = trained.find_latest_ckpt(pull=True)
        assert path is not None
        state = torch.load(path, weights_only=False)
        assert 'state_dict' in state

    def test_tensorboard_logs_were_written(self, trained):
        runs = os.listdir(trained.dirpath('logs'))
        assert any(r.startswith('run_') for r in runs), runs

    def test_from_scratch_ignores_its_own_latest(self, trained):
        fresh = type(trained)(**{**trained.dfn, 'tag': 'toy',
                                 'spec': {**trained.spec, 'from_scratch': True}})
        assert fresh.var.from_scratch is True
        # _resume_plan_ consults var.ckpt_builder only; with neither, nothing to resume.
        assert fresh._resume_plan_() is None

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
        still = make_still(tmp_path, ckpt_builder=source)
        ckpt, own_it, warm_start = still._resolve_resume_ckpt_()
        assert warm_start is True
        assert own_it is False
        assert ckpt == source.find_latest_ckpt(pull=True)

    def test_a_path_is_a_full_resume(self, source, tmp_path):
        """The escape hatch: pointing at the file, not the block, is how you
        ask for the optimizer state back."""
        path = source.find_latest_ckpt(pull=True)
        still = make_still(tmp_path, ckpt_builder=path)
        ckpt, own_it, warm_start = still._resolve_resume_ckpt_()
        assert warm_start is False
        assert ckpt == path

    def test_this_runs_own_checkpoint_wins(self, source, tmp_path):
        """Once it has trained, a restart continues *this* run -- the warm-start
        source is history, not something to fall back to on every launch."""
        still = make_still(tmp_path, ckpt_builder=source)
        still.build()
        ckpt, own_it, warm_start = still._resolve_resume_ckpt_()
        assert warm_start is False
        assert own_it is True

    def test_a_source_with_no_checkpoints_raises(self, tmp_path):
        untrained = make_still(tmp_path / 'untrained')
        still = make_still(tmp_path / 'still', ckpt_builder=untrained)
        with pytest.raises(FileNotFoundError, match='no checkpoints'):
            still._resolve_resume_ckpt_()

    def test_the_banner_says_weights_only(self, source, tmp_path):
        plan = make_still(tmp_path, ckpt_builder=source)._resume_plan_()
        assert 'warm start, weights only' in plan
        assert source.key in plan

    def test_a_full_resume_plan_is_the_checkpoint_itself(self, source, tmp_path):
        path = source.find_latest_ckpt(pull=True)
        assert make_still(tmp_path, ckpt_builder=path)._resume_plan_() == path

    def test_the_divergence_report_is_skipped_for_a_warm_start(self, source, tmp_path, caplog):
        """The plan names the block, the checkpoint is a file -- comparing the
        two would cry wolf on every warm start."""
        still = make_still(tmp_path, ckpt_builder=source)
        still._report_resume_divergence_(
            source.find_latest_ckpt(pull=True), still._resume_plan_(), warm_start=True,
        )
        assert 'NOT the one in the banner' not in caplog.text

    def test_it_trains_its_own_schedule_from_step_zero(self, source, tmp_path):
        """The behaviour the rest of this class is about, end to end: the
        warm-started run does its own 2 epochs rather than inheriting the
        source's exhausted epoch counter and stopping instantly."""
        still = make_still(tmp_path, ckpt_builder=source)
        still.build()
        assert still.valid() is True
        steps = {still._ckpt_step_(n)
                 for n in os.listdir(still.dirpath('ckpts')) if n.endswith('.ckpt')}
        source_steps = {source._ckpt_step_(n)
                        for n in os.listdir(source.dirpath('ckpts')) if n.endswith('.ckpt')}
        assert steps == source_steps, "should have run its own schedule, from 0"

    def test_the_weights_are_the_sources(self, source, tmp_path):
        """...and they are the source's weights, not a fresh init."""
        still = make_still(tmp_path, ckpt_builder=source)
        model = still.lightning_module
        key = next(k for k in model.state_dict() if k.endswith('.weight'))
        before = model.state_dict()[key].clone()
        still._load_weights_only_(model, source.find_latest_ckpt(pull=True), why='test')
        after = model.state_dict()[key]
        # map_location='cpu' in the comparison too: the source ran through a
        # real Trainer, so on a machine with a GPU its checkpoint holds CUDA
        # tensors, while `model` has never been fitted and is still on the
        # host. Without this the assert is a device mismatch rather than a
        # comparison -- and it passes on CPU-only CI, which is where it hid.
        expected = torch.load(
            source.find_latest_ckpt(pull=True),
            map_location='cpu', weights_only=False,
        )['state_dict'][key]
        assert torch.equal(after, expected)
        assert not torch.equal(after, before)


class TestLoadWeightsOnly:
    """The load itself -- what ``strict=False`` will and will not catch."""

    @pytest.fixture
    def still(self, tmp_path):
        return make_still(tmp_path)

    def _ckpt_(self, tmp_path, payload):
        path = tmp_path / 'hand.ckpt'
        torch.save(payload, path)
        return str(path)

    def test_a_checkpoint_without_a_state_dict_says_so(self, still, tmp_path):
        path = self._ckpt_(tmp_path, {'epoch': 3})
        with pytest.raises(KeyError, match='state_dict'):
            still._load_weights_only_(still.lightning_module, path, why='test')

    def test_a_checkpoint_of_another_architecture_is_refused(self, still, tmp_path):
        """strict=False silently ignores names it doesn't know, so a warm start
        pointed at the wrong model loads *nothing* and trains from random init
        while the banner claims otherwise."""
        path = self._ckpt_(tmp_path, {'state_dict': {'nothing.like.it': torch.zeros(3)}})
        with pytest.raises(ValueError, match='nothing loaded'):
            still._load_weights_only_(still.lightning_module, path, why='test')

    def test_the_checkpoint_does_not_choose_the_device(self, still, tmp_path,
                                                       monkeypatch):
        """Staged on the host, whatever device saved it.

        A Trainer that fitted on ``cuda:0`` saves CUDA tensors. Honouring that
        makes the checkpoint unloadable anywhere else -- ``torch.load`` raises
        on a CPU-only box, and raises an invalid-device error on a machine with
        fewer GPUs -- and where it does work it holds a second, GPU-resident
        copy of the whole state dict while the model that is about to train
        competes for the same memory.

        Asserted on the call rather than the outcome because the outcome is
        invisible on any single machine: on this one a CUDA checkpoint loads
        either way, and on CPU-only CI there is no CUDA checkpoint to make one.
        """
        model = still.lightning_module
        path = self._ckpt_(tmp_path, {'state_dict': dict(model.state_dict())})
        seen = {}
        real_load = torch.load

        def spy(f, *args, **kwargs):
            seen.update(kwargs)
            return real_load(f, *args, **kwargs)

        monkeypatch.setattr(torch, 'load', spy)
        still._load_weights_only_(model, path, why='test')
        assert seen.get('map_location') == 'cpu', seen

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
    def test_a_checkpoint_of_cuda_tensors_loads_onto_the_host(self, still, tmp_path):
        """The behaviour the call above buys, where the hardware can show it."""
        model = still.lightning_module
        state = {k: v.cuda() for k, v in model.state_dict().items()}
        path = self._ckpt_(tmp_path, {'state_dict': state})
        still._load_weights_only_(model, path, why='test')
        assert all(p.device.type == 'cpu' for p in model.parameters())

    def test_a_partial_match_is_allowed(self, still, tmp_path):
        """A changed head is the reason this loads with strict=False at all."""
        model = still.lightning_module
        state = dict(model.state_dict())
        state['some.extra.head'] = torch.zeros(3)
        path = self._ckpt_(tmp_path, {'state_dict': state})
        still._load_weights_only_(model, path, why='test')   # does not raise


class TestCkptStep:
    @pytest.mark.parametrize('name,step', [
        ('epoch=003-step=0001234.ckpt', 1234),
        ('epoch=000-step=step=0000008.ckpt', 8),   # Lightning's doubled form
        ('no-step-here.ckpt', -1),
    ])
    def test_parses_the_step_out_of_the_filename(self, name, step):
        assert Still._ckpt_step_(name) == step

    def test_sorts_numerically_not_lexically(self):
        names = ['step=9.ckpt', 'step=10.ckpt', 'step=100.ckpt']
        assert sorted(names, key=Still._ckpt_step_) == names


# ═══════════════════════════════════════════════════════════════════════
#  LightningBuilder
# ═══════════════════════════════════════════════════════════════════════

class TestLightning:
    def test_valid_reflects_nested_var_blocks(self, tmp_path):
        class Upstream(Datablock):
            VERSION = 1
            TOPICS = {'thing': 'thing.txt'}

            def __build__(self):
                with self.fs.open(self.path('thing', ensure_dirpath=True), 'w') as f:
                    f.write('x')
                return self

        class Wrapper(LightningBuilder):
            VERSION = 1

            @dataclass
            class VAR(LightningBuilder.VAR):
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

        class Wrapper(LightningBuilder):
            VERSION = 1

            @dataclass
            class VAR(LightningBuilder.VAR):
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
        class Wrapper(LightningBuilder):
            VERSION = 1

            def __lightning_module__(self):
                return ToyLightning(ToyModel())

        with pytest.raises(NotImplementedError, match="unreachable"):
            Wrapper(url=str(tmp_path), tag='w').__build__()

    def test_the_module_is_cached(self, tmp_path):
        class Wrapper(LightningBuilder):
            VERSION = 1

            def __lightning_module__(self):
                return ToyLightning(ToyModel())

        w = Wrapper(url=str(tmp_path), tag='w')
        assert w.lightning_module is w.lightning_module

    def test_an_unimplemented_module_says_so(self, tmp_path):
        class Wrapper(LightningBuilder):
            VERSION = 1

        with pytest.raises(NotImplementedError, match="__lightning_module__"):
            Wrapper(url=str(tmp_path), tag='w').lightning_module

    def test_the_lightning_builder_in_var_supplies_the_still_its_module(self, tmp_path):
        still = make_still(tmp_path, learning_rate=0.123)
        assert still.lightning_module.lr == 0.123

    def test_the_model_builder_in_var_supplies_the_still_its_model(self, tmp_path):
        model = make_still(tmp_path, width=4).model()
        assert model.net[0].in_features == 4


# ═══════════════════════════════════════════════════════════════════════
#  Weights
# ═══════════════════════════════════════════════════════════════════════

class TestWeights:
    def test_no_ckpt_is_trivially_valid_and_has_no_local_path(self, tmp_path):
        w = Weights(url=str(tmp_path), tag='none', spec=dict(ckpt=None))
        assert w.valid() is True
        w.build()
        assert w.path_local() is None

    def test_fetches_persists_and_hands_back_a_local_path(self, tmp_path):
        src = tmp_path / 'src.pt'
        torch.save({'k': torch.zeros(3)}, src)
        w = Weights(url=str(tmp_path / 'store'), tag='w',
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
        w = Weights(url=str(tmp_path / 'store'), tag='w', spec=dict(ckpt=str(src)))
        w.build()
        with open(w.path('weights'), 'r+b') as f:
            f.truncate(16)
        assert w.valid() is False

    def test_source_url_is_the_extension_point(self, tmp_path):
        src = tmp_path / 'real.pt'
        torch.save({'k': 1}, src)

        class Registry(Weights):
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
