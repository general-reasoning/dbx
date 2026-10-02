"""`read_checkpoint`: a checkpoint whose pickled extras name a class since removed still yields its weights."""
import sys
import types

import pytest
import torch

from dbx.journals import Datalog
from dbx.stills import read_checkpoint


def test_a_class_gone_since_the_save_reads_as_a_stand_in(tmp_path, monkeypatch):
    mod = types.ModuleType('gone_since')
    class Logger:                                   # pickled into the hyperparameters, then renamed away
        def __init__(self):
            self.name = 'old'
    Logger.__module__, Logger.__qualname__ = 'gone_since', 'Logger'
    mod.Logger = Logger
    monkeypatch.setitem(sys.modules, 'gone_since', mod)
    path = tmp_path / 'run.ckpt'
    weights = {'model.w': torch.arange(4.0)}
    torch.save({'state_dict': weights, 'hyper_parameters': {'log': Logger()}}, path)
    monkeypatch.delattr(mod, 'Logger')

    with pytest.raises(AttributeError):
        torch.load(path, map_location='cpu', weights_only=False)
    ckpt = read_checkpoint(path, log=Datalog())
    assert torch.equal(ckpt['state_dict']['model.w'], weights['model.w'])
    assert type(ckpt['hyper_parameters']['log']).__name__ == 'Logger'
