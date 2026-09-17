"""test_stills.py — Still, the builder blocks it trains from, Weights.

The training-loop tests really do train: a 72-parameter linear model over a
synthetic dataset for two epochs, which is fast and is the only way to pin what
``__build__`` actually leaves behind (checkpoints at both cadences, the
``done`` topic, a resumable latest).
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
    ChunkShuffleSampler,
    ResumableDataLoader,
    CheckpointBuilder,
    CheckpointPath,
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
        max_epochs=2, batch_size=16, chunk_shuffle_size=32,
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
            'train_val_split', 'dataset_seed', 'chunk_shuffle_size',
            'from_scratch', 'reset_optimizer_state', 'max_epochs',
            'max_training_steps', 'train_log_every_n_steps', 'val_every_n_steps',
            'val_max_batches', 'val_shuffle', 'val_seed', 'gradient_clip_val',
            'gradient_clip_algorithm', 'ckpt_every_n_steps', 'ckpt_every_n_epochs',
            'precision', 'matmul_precision', 'accumulate_grad_batches',
            'batch_size', 'still_seed',
        ]

    @pytest.mark.pinned
    def test_the_version_is_a_ledger_of_key_moving_changes(self):
        """Every change that moves every still's key bumps VERSION, so the path
        records it instead of artifacts drifting silently.

            2 -- the builder VAR replaced the cfg_ surface
            3 -- `done` became a topic of its own, changing TOPICS and so the
                 signature, where completion had been a `_COMPLETE` file
                 inside `ckpts`
            4 -- VAR.block_shuffle_size became chunk_shuffle_size, after the
                 ChunkShuffleSampler it configures -- a VAR field name is in
                 the signature

        Do not edit this to agree with a bump you made. Add the line saying
        what moved -- that is the point of the pin.
        """
        assert Still.VERSION == 4


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
        train, _ = make_still(tmp_path, chunk_shuffle_size=32).dataloaders()
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
        assert os.path.exists(trained.path('done', local=True))

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
    ``done`` topic is written, and an untrained model sits at a key that
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
    """The load itself -- what ``strict=False`` will and will not catch.

    Every still here asks for the lenient load explicitly: ``strict_loading``
    defaults to True now, and under it a mismatch raises long before reaching
    the checks these tests are about.
    """

    @pytest.fixture
    def still(self, tmp_path):
        return ToyStill(url=str(tmp_path), tag='toy', spec=toy_builders(tmp_path),
                        num_workers=0, strict_loading=False)

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
        """A changed head is the reason strict_loading=False exists at all."""
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


# ═══════════════════════════════════════════════════════════════════════
#  CheckpointPath — a warm-start source named by path
# ═══════════════════════════════════════════════════════════════════════

class TestCheckpointPath:
    """A checkpoint addressed as a file, for when its Still can no longer be built.

    ``ckpt_builder=<Still>`` needs that Still reconstructed, which stops being
    possible after a ``VERSION`` bump or a module move -- while the checkpoint
    itself is still sitting there and still holds the weights. This names it.
    """

    @pytest.fixture(scope='class')
    def source(self, tmp_path_factory):
        still = make_still(tmp_path_factory.mktemp('cp_source'))
        still.build()
        return still

    @pytest.fixture
    def ckpt(self, source):
        return source.find_latest_ckpt(pull=True)

    def make(self, root, path):
        return CheckpointPath(url=str(root), tag='cp', spec=dict(ckpt_path=path))

    # Identity ──────────────────────────────────────────────────────

    @pytest.mark.pinned
    def test_the_path_is_in_the_identity(self, tmp_path):
        """The reason this class exists rather than a bare string.

        A run warm-started from step 319000 has to be a different artifact
        from the same run warm-started from step 318000 -- they hold
        different weights. If these two ever hash alike, the second run
        silently lands on the first one's key and reads back its checkpoints.
        """
        a = self.make(tmp_path, '/ckpts/epoch=029-step=0319000.ckpt')
        b = self.make(tmp_path, '/ckpts/epoch=029-step=0318000.ckpt')
        assert a.hash != b.hash

    def test_the_same_path_is_the_same_block(self, tmp_path):
        p = '/ckpts/epoch=029-step=0319000.ckpt'
        assert self.make(tmp_path, p).hash == self.make(tmp_path, p).hash

    # find_latest_ckpt ──────────────────────────────────────────────

    def test_it_answers_with_the_path_it_was_given(self, tmp_path, ckpt):
        assert self.make(tmp_path, ckpt).find_latest_ckpt() == ckpt

    def test_no_path_is_no_checkpoint(self, tmp_path):
        assert self.make(tmp_path, None).find_latest_ckpt() is None
        assert self.make(tmp_path, None).valid() is False

    def test_pull_of_a_local_file_hands_it_back_as_it_stands(self, tmp_path, ckpt):
        """No copy: we neither own the file nor would clean the duplicate up."""
        assert self.make(tmp_path, ckpt).find_latest_ckpt(pull=True) == ckpt

    def test_pull_of_a_missing_file_raises(self, tmp_path):
        block = self.make(tmp_path, str(tmp_path / 'nope.ckpt'))
        with pytest.raises(FileNotFoundError, match='no checkpoint at'):
            block.find_latest_ckpt(pull=True)

    def test_pull_of_something_that_is_not_a_checkpoint_raises(self, tmp_path):
        """A truncated download must not reach torch.load as if it were fine."""
        junk = tmp_path / 'junk.ckpt'
        junk.write_text('not a zip archive')
        with pytest.raises(RuntimeError, match='not a readable checkpoint'):
            self.make(tmp_path, str(junk)).find_latest_ckpt(pull=True)

    # valid ─────────────────────────────────────────────────────────

    def test_valid_is_whether_the_file_is_there(self, tmp_path, ckpt):
        assert self.make(tmp_path, ckpt).valid() is True
        assert self.make(tmp_path, str(tmp_path / 'gone.ckpt')).valid() is False

    def test_the_step_is_read_off_the_filename(self, tmp_path):
        assert self.make(tmp_path, '/c/epoch=029-step=0319000.ckpt').ckpt_step == 319000
        assert self.make(tmp_path, '/c/handwritten.ckpt').ckpt_step == -1

    # What a Still does with one ────────────────────────────────────

    @pytest.mark.pinned
    def test_a_still_treats_it_as_a_warm_start(self, tmp_path, ckpt):
        """It is a CheckpointBuilder, so weights only, at step 0.

        The same path passed as a bare string is a full resume instead --
        which restores the source's epoch counter, and so can leave a run
        that has trained nothing sitting at a key claiming it finished.
        """
        still = make_still(tmp_path, ckpt_builder=self.make(tmp_path, ckpt))
        got, own_it, warm_start = still._resolve_resume_ckpt_()
        assert warm_start is True
        assert own_it is False
        assert got == ckpt

    def test_the_string_form_is_still_a_full_resume(self, tmp_path, ckpt):
        """The distinction the block exists to make -- it has to keep holding."""
        still = make_still(tmp_path, ckpt_builder=ckpt)
        _, _, warm_start = still._resolve_resume_ckpt_()
        assert warm_start is False

    def test_the_banner_says_weights_only(self, tmp_path, ckpt):
        block = self.make(tmp_path, ckpt)
        plan = make_still(tmp_path, ckpt_builder=block)._resume_plan_()
        assert 'warm start, weights only' in plan
        assert block.key in plan

    def test_the_still_hash_moves_with_the_checkpoint(self, tmp_path, ckpt):
        a = make_still(tmp_path, ckpt_builder=self.make(tmp_path, ckpt))
        b = make_still(tmp_path, ckpt_builder=self.make(tmp_path, ckpt + '.other'))
        assert a.hash != b.hash


# ═══════════════════════════════════════════════════════════════════════
#  Devices, strategy, check_run, strict loading
# ═══════════════════════════════════════════════════════════════════════

def _toy(root, **kw):
    return ToyStill(url=str(root), tag='toy', spec=toy_builders(root), num_workers=0, **kw)


def _trainer_kwargs(root, **kw):
    return _toy(root, **kw).trainer_kwargs(ckpts_dir='/tmp/x', callbacks=[], tb_logger=None)


class TestDevices:
    """``devices`` names the devices; ``n_devices`` only ever counted them."""

    @pytest.mark.parametrize('spec, accelerator, devices', [
        (['cuda'],            'cuda', 1),
        (['cuda', 'cuda'],    'cuda', 2),
        (['cuda:1', 'cuda:2'], 'cuda', [1, 2]),
        (['cuda:2', 'cuda:1'], 'cuda', [2, 1]),
        (['cpu'],             'cpu',  1),
        (['cpu', 'cpu'],      'cpu',  2),
        (['gpu'],             'gpu',  1),
        ('cpu',               'cpu',  1),
    ])
    def test_it_resolves(self, tmp_path, spec, accelerator, devices):
        k = _trainer_kwargs(tmp_path, devices=spec)
        assert (k['accelerator'], k['devices']) == (accelerator, devices)

    @pytest.mark.parametrize('spec', [1, 4, -1, 'auto'])
    def test_lightnings_own_vocabulary_passes_through(self, tmp_path, spec):
        """Anything Trainer(devices=) already took keeps working, accelerator
        left to Lightning's 'auto' as before."""
        k = _trainer_kwargs(tmp_path, devices=spec)
        assert k['devices'] == spec
        assert 'accelerator' not in k

    def test_n_devices_still_works(self, tmp_path):
        """The old surface is in recorded dfns; it cannot stop working."""
        k = _trainer_kwargs(tmp_path, n_devices=4)
        assert k['devices'] == 4 and 'accelerator' not in k

    def test_the_default_is_one_device_and_no_accelerator(self, tmp_path):
        k = _trainer_kwargs(tmp_path)
        assert k['devices'] == 1
        assert 'accelerator' not in k and 'strategy' not in k

    @pytest.mark.parametrize('spec, why', [
        (['cpu', 'cuda'],    'one Trainer runs on one accelerator'),
        (['cuda:0', 'cuda'], 'all indexed or all bare'),
        (['cpu:0', 'cpu:1'], 'cpu has no index'),
        (['cuda:0', 'cuda:0'], 'named twice'),
        (['cuda:x'],         'index is not an integer'),
        ([],                 'empty'),
        ([0, 1],             'not strings'),
    ])
    def test_it_refuses(self, tmp_path, spec, why):
        with pytest.raises(ValueError):
            _trainer_kwargs(tmp_path, devices=spec)

    def test_the_two_surfaces_cannot_both_be_given(self, tmp_path):
        with pytest.raises(ValueError, match='not both'):
            _toy(tmp_path, devices=['cpu'], n_devices=2)

    def test_a_strategy_is_passed_when_asked_for(self, tmp_path):
        k = _trainer_kwargs(tmp_path, devices=['cpu', 'cpu'], strategy='ddp_spawn')
        assert k['strategy'] == 'ddp_spawn'

    @pytest.mark.pinned
    def test_none_of_it_reaches_the_identity(self, tmp_path):
        """Where a run executes is operational. If any of this moved the hash,
        the same run on one GPU and on four would be two artifacts."""
        base = _toy(tmp_path).hash
        for kw in (dict(n_devices=8), dict(devices=['cpu', 'cpu']),
                   dict(devices=['cuda:3']), dict(strategy='ddp_spawn'),
                   dict(check_run=True)):
            assert _toy(tmp_path, **kw).hash == base, kw


class TestCheckRun:
    """One batch through, to find the bug before the 30 epochs."""

    def test_it_reaches_the_trainer(self, tmp_path):
        """Ours is check_run; Lightning's own name for it is fast_dev_run."""
        assert _trainer_kwargs(tmp_path, check_run=True)['fast_dev_run'] is True
        assert _trainer_kwargs(tmp_path, check_run=3)['fast_dev_run'] == 3

    def test_it_is_absent_by_default(self, tmp_path):
        assert 'fast_dev_run' not in _trainer_kwargs(tmp_path)

    @pytest.mark.pinned
    def test_it_does_not_mark_the_block_complete(self, tmp_path):
        """The whole hazard of a smoke test that goes through __build__.

        fit() returning is what __build__ takes as "trained", so without this
        one batch would write `done`, valid() would agree, and the real
        build afterwards would skip the block entirely -- leaving an untrained
        model at a key that claims otherwise.
        """
        still = _toy(tmp_path, check_run=True)
        still.build()
        assert still.valid() is False
        assert not os.path.exists(still.path('done', local=True))

    def test_without_it_the_block_is_built(self, tmp_path):
        """The other half: the guard must not be suppressing ordinary runs."""
        still = _toy(tmp_path)
        still.build()
        assert still.valid() is True


class TestStrictLoading:
    """``strict_loading`` on a weights-only load: does this checkpoint fit?"""

    @pytest.fixture(scope='class')
    def ckpt(self, tmp_path_factory):
        source = make_still(tmp_path_factory.mktemp('strict_src'))
        source.build()
        return source.find_latest_ckpt(pull=True)

    def _load(self, root, ckpt, *, strict, mutate=None):
        still = _toy(root, strict_loading=strict)
        model = still.var.lightning_builder.lightning_module
        if mutate:
            mutate(model)
        return still._load_weights_only_(model, ckpt, why='test')

    @pytest.mark.parametrize('strict', [False, True])
    def test_matching_weights_load_either_way(self, tmp_path, ckpt, strict):
        """strict=True must not reject a checkpoint that does fit -- otherwise
        it is useless as the check it exists to be."""
        self._load(tmp_path, ckpt, strict=strict)

    def test_strict_catches_an_architecture_that_does_not_fit(self, tmp_path, ckpt):
        with pytest.raises(RuntimeError, match='state_dict'):
            self._load(tmp_path, ckpt, strict=True, mutate=_add_a_head_)

    def test_lenient_partial_loads_and_is_no_longer_the_default(self, tmp_path, ckpt):
        """A warm start into a changed head has missing keys by construction,
        which is what strict_loading=False is for -- now that it has to be
        asked for, since the assertion a warm start makes should have to hold
        unless you say otherwise."""
        assert _toy(tmp_path).strict_loading is True
        self._load(tmp_path, ckpt, strict=False, mutate=_add_a_head_)

    def test_a_wholly_unrelated_checkpoint_is_refused_even_when_lenient(self, tmp_path, ckpt):
        """strict=False still will not accept a checkpoint that shares no
        parameter name at all -- that is random init wearing a banner."""
        still = _toy(tmp_path, strict_loading=False)
        model = still.var.lightning_builder.lightning_module
        with pytest.raises(ValueError, match='nothing loaded'):
            still._load_weights_only_(model, _unrelated_ckpt_(tmp_path), why='test')


def _add_a_head_(model):
    """A parameter this model has and the checkpoint does not."""
    model.extra_head = torch.nn.Linear(4, 4)


def _unrelated_ckpt_(tmp_path):
    path = os.path.join(str(tmp_path), 'unrelated.ckpt')
    torch.save({'state_dict': {'nothing.to.do.with.it': torch.zeros(2)}}, path)
    return path


class TestDoneTopic:
    """Completion is a topic, not a file dbx knows nothing about."""

    def test_it_is_a_topic(self, tmp_path):
        assert Still.TOPICS['done'] == 'done'
        assert 'done' in _toy(tmp_path).topics()

    def test_valid_reads_it(self, tmp_path):
        still = _toy(tmp_path)
        assert still.valid() is False
        still.UNSAFE_done(OVERRIDE=True)
        assert still.valid() is True

    def test_a_finished_run_writes_it(self, tmp_path):
        still = _toy(tmp_path)
        still.build()
        assert os.path.exists(still.path('done', local=True))
        assert still.valid() is True

    @pytest.mark.pinned
    def test_clearing_ckpts_clears_done(self, tmp_path):
        """The one coupling the topic split costs.

        `done` asserts that the checkpoints beside it are a finished run. Drop
        those and leave it, and `valid` reports a trained model with no weights
        behind it -- and `build_tree` skips the block that would rebuild them.
        """
        still = _toy(tmp_path)
        still.build()
        assert still.valid() is True
        still.UNSAFE_clear('ckpts', OVERRIDE=True)
        assert still.valid() is False

    def test_clearing_done_alone_leaves_the_checkpoints(self, tmp_path):
        """Not the reverse coupling: this is how a run marked complete too
        early is reopened without throwing its weights away."""
        still = _toy(tmp_path)
        still.build()
        ckpt = still.find_latest_ckpt(pull=True)
        still.UNSAFE_clear('done', OVERRIDE=True)
        assert still.valid() is False
        assert still.find_latest_ckpt(pull=True) == ckpt

    def test_clearing_everything_clears_done(self, tmp_path):
        still = _toy(tmp_path)
        still.build()
        still.UNSAFE_clear(OVERRIDE=True)
        assert still.valid() is False


class TestResumableDataLoaderUnderDDP:
    """The loader must find its sampler's state through Lightning's wrapper.

    Under DDP, Lightning replaces the sampler with a DistributedSamplerWrapper
    so each rank draws a disjoint subset. That wrapper has no `state_dict`, so
    a loader that asked `self.sampler` directly raised at the first checkpoint
    -- after training had started, which is the expensive place to find out.
    """

    def _loader(self, sampler=None):
        ds = torch.utils.data.TensorDataset(torch.arange(64))
        kw = {'sampler': sampler} if sampler is not None else {}
        return ResumableDataLoader(ds, batch_size=4, **kw)

    def test_a_plain_sampler_is_unchanged(self):
        sampler = ChunkShuffleSampler(64, 8, seed=1)
        assert self._loader(sampler).state_dict() == sampler.state_dict()

    @pytest.mark.pinned
    def test_a_ddp_wrapped_sampler_is_reached_through_the_wrapper(self):
        """The bug this exists for: the state is the inner sampler's."""
        from lightning.fabric.utilities.distributed import DistributedSamplerWrapper

        inner = ChunkShuffleSampler(64, 8, seed=1)
        inner.set_epoch(3)
        wrapped = DistributedSamplerWrapper(inner, num_replicas=2, rank=0)
        assert self._loader(wrapped).state_dict() == {'epoch': 3, 'consumed': 0}

    def test_a_loader_with_no_stateful_sampler_says_so(self):
        """Empty, not an exception -- a loader may legitimately have none."""
        assert self._loader().state_dict() == {}
        self._loader().load_state_dict({})          # does not raise

    def test_a_round_trip_through_the_wrapper(self):
        from lightning.fabric.utilities.distributed import DistributedSamplerWrapper

        inner = ChunkShuffleSampler(64, 8, seed=1)
        loader = self._loader(DistributedSamplerWrapper(inner, num_replicas=2, rank=0))
        loader.load_state_dict({'epoch': 7, 'consumed': 0})
        assert inner.epoch == 7
        assert loader.state_dict() == {'epoch': 7, 'consumed': 0}


class TestCheckRunWritesNothing:
    """A check run leaves no artifact behind -- not `done`, not a checkpoint."""

    def _callbacks(self, root, **kw):
        still = _toy(root, **kw)
        return [type(c).__name__ for c in still.callbacks(ckpts_dir=str(root))]

    def test_an_ordinary_run_installs_the_checkpointers(self, tmp_path):
        names = self._callbacks(tmp_path)
        assert '_StepCheckpoint' in names or '_EpochCheckpoint' in names

    @pytest.mark.pinned
    def test_a_check_run_installs_none_of_them(self, tmp_path):
        """fast_dev_run stops after one batch -- which is also an epoch END, so
        ckpt_every_n_epochs=1 fires and a check run saved and uploaded a
        multi-GB checkpoint to prove a batch flows.
        """
        names = self._callbacks(tmp_path, check_run=True)
        assert not any('Checkpoint' in n for n in names)

    def test_the_cite_callback_survives(self, tmp_path):
        """Suppressing checkpoints must not suppress the identity record."""
        assert '_LogCiteOnStart' in self._callbacks(tmp_path, check_run=True)

    def test_a_check_run_writes_no_checkpoint_end_to_end(self, tmp_path):
        still = _toy(tmp_path, check_run=True)
        still.build()
        ckpts = still.dirpath('ckpts', local=True)
        assert not [f for f in os.listdir(ckpts) if f.endswith('.ckpt')]
        assert still.valid() is False


class TestCheckpointIsWrittenOnceUnderDDP:
    """Only rank 0 uploads, frees and journals a checkpoint.

    trainer.save_checkpoint is collective and every rank must call it, but it
    writes the file on rank 0 alone. Without the guard after it, every rank
    pushed the same multi-GB blob to the same remote path while `free_src=True`
    deleted the local copy from under the others -- an upload that stalled at
    0% and never returned.
    """

    class _FakeTrainer:
        def __init__(self, rank):
            self.global_rank = rank
            self.saved = []

        def save_checkpoint(self, path):
            self.saved.append(path)

    def _save_fn(self, still, tmp_path, monkeypatch):
        """The `_save` closure out of callbacks(), with its effects recorded."""
        effects = {'uploaded': [], 'journalled': []}
        monkeypatch.setattr(still, 'push',
                            lambda *a, **k: effects['uploaded'].append(a[0]))
        monkeypatch.setattr(still, 'write_journal_entry',
                            lambda **k: effects['journalled'].append(k.get('event')))
        monkeypatch.setattr(type(still), 'is_local_fs', property(lambda s: False))
        cbs = still.callbacks(ckpts_dir=str(tmp_path))
        epoch_cb = next(c for c in cbs if 'Epoch' in type(c).__name__)
        return epoch_cb, effects

    @pytest.mark.pinned
    def test_only_rank_zero_uploads_and_journals(self, tmp_path, monkeypatch):
        # ckpt_every_n_epochs is a VAR field, so it goes in the spec -- as a
        # constructor kwarg it lands in `parameters` and the default (5) still
        # applies, and on_train_epoch_end returns before saving anything.
        still = ToyStill(url=str(tmp_path), tag='toy', num_workers=0,
                         spec=dict(toy_builders(tmp_path), ckpt_every_n_epochs=1))
        cb, effects = self._save_fn(still, tmp_path, monkeypatch)

        for rank in (0, 1, 2, 3):
            trainer = self._FakeTrainer(rank)
            trainer.current_epoch, trainer.global_step = 0, 1
            cb.on_train_epoch_end(trainer, None)
            # Every rank participates in the collective save.
            assert trainer.saved, f"rank {rank} skipped the collective save"

        assert len(effects['uploaded']) == 1, effects['uploaded']
        assert len(effects['journalled']) == 1, effects['journalled']
