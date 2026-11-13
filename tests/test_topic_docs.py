"""Topics that say what they hold: `doc=`, a directory's declared files and patterns, `over=`, `DATAFRAME`.

A topic's filename says where its data is and nothing about what it holds. The
markers can say more: `DATADICT` and `DATAFRAME` the keys or columns of a file,
`DATADIR` the files and directories under it -- by name, or by a pattern such as
`'{scene}.npz'`, any number of entries named so -- and every marker a `doc=`, the
sentence a reader would otherwise go and find. `over=` says what a pattern's
placeholders range over: a `VAR` reference (`VAR.splits`, recorded, never read
at declaration), literal values, or words.

The shape renders into the identity; the documentation does not. `doc=` and
`over=` reach the journal -- `render_topics(doc=True)`, read back by
`literal_topics` -- and `pretty.topics(doc=True)`, never the hash: rewording a
doc re-keys nothing. The note before it rendered, and still does, for the
declarations built under it.
"""
import copy
import os
import pickle
from dataclasses import dataclass

import pytest

import dbx
from dbx.datablocks import (
    DATADICT,
    DATADIR,
    DATAFILE,
    DATAFRAME,
    VAR,
    Datablock,
    literal_topics,
    render_topics,
)


def _block(topics, **var):
    """A block class declaring *topics*, its VAR the fields *var* with their defaults."""
    namespace = {'TOPICS': topics, '__module__': __name__}
    fields_ = {name: dataclass_field for name, dataclass_field in var.items()}
    VARCLS = dataclass(type('VAR', (Datablock.VAR,), {'__annotations__': {k: type(v) for k, v in fields_.items()},
                                                    **fields_}))
    namespace['VAR'] = VARCLS
    return type('Block', (Datablock,), namespace)


def _hash(topics, tmp_path, **var):
    return _block(topics, **var)(datalake=str(tmp_path)).hash


# --------------------------------------------------------------------------------------------- #
# doc= reaches no identity
# --------------------------------------------------------------------------------------------- #

@pytest.mark.pinned
@pytest.mark.parametrize('bare, documented', [
    (DATAFILE('c.yaml'), DATAFILE('c.yaml', doc='what the build resolved')),
    (DATADICT('c.yaml', n='int'), DATADICT('c.yaml', n='int', doc='the counts')),
    (DATAFRAME('s.parquet', a='int'), DATAFRAME('s.parquet', a='int', doc='one row per scene')),
    (DATADIR, DATADIR(doc='the logs')),
    (DATADIR({'{split}': DATAFILE('{scene}.npz')}),
     DATADIR({'{split}': DATADIR(DATAFILE('{scene}.npz', doc='one scene'), doc='a split')},
             over={'split': VAR.splits, 'scene': 'the scenes'}, doc='the scenes, by split')),
])
def test_documentation_never_reaches_the_hash(tmp_path, bare, documented):
    assert repr(bare) == repr(documented)
    assert _hash({'t': bare}, tmp_path, splits=('a',)) == _hash({'t': documented}, tmp_path, splits=('a',))


def test_the_note_still_reaches_it(tmp_path):
    assert _hash({'t': DATAFILE('c.yaml', 'a note')}, tmp_path) != _hash({'t': DATAFILE('c.yaml')}, tmp_path)


@pytest.mark.pinned
def test_declarations_from_before_render_as_they_did():
    assert repr(DATADIR('meta', {'videos': ('chunk-000',)})) == "DATADIR('meta', {'videos': ('chunk-000',)})"
    assert repr(DATADIR('one PNG per frame')) == "DATADIR('one PNG per frame')"
    assert repr(DATADIR('meta', note='a lerobot dataset')) == "DATADIR('meta', note='a lerobot dataset')"
    assert repr(DATAFILE('rows.csv', 'one row per frame')) == "DATAFILE('rows.csv', 'one row per frame')"
    assert repr(DATADICT('m.json', rows='int', run=dict(id='str'))) == \
        "DATADICT('m.json', rows='int', run=dict(id='str'))"


def test_a_doc_is_prose():
    for bad in ('', '  ', 3):
        with pytest.raises(TypeError, match='doc'):
            DATAFILE('c.yaml', doc=bad)


# --------------------------------------------------------------------------------------------- #
# A directory's children: files, levels, patterns
# --------------------------------------------------------------------------------------------- #

def test_a_level_renders_the_same_however_it_is_spelled(tmp_path):
    spelled = [DATADIR({'a': ('x', DATAFILE('f.json'))}),
               DATADIR({'a': DATADIR('x', DATAFILE('f.json'))}),
               DATADIR({'a': DATADIR('x', DATAFILE('f.json', doc='a file'), doc='a level')})]
    assert len({repr(m) for m in spelled}) == 1
    assert len({_hash({'t': m}, tmp_path) for m in spelled}) == 1


def test_a_file_among_the_children_is_its_marker_and_a_level_needs_a_name():
    m = DATADIR(DATAFILE('index.json'), {'{split}': DATAFILE('{scene}.npz')})
    assert repr(m) == "DATADIR(DATAFILE('index.json'), {'{split}': (DATAFILE('{scene}.npz'),)})"
    with pytest.raises(TypeError, match='needs a name'):
        DATADIR(DATADIR('x'))
    with pytest.raises(TypeError, match='doc=, not a note'):
        DATADIR(DATAFILE('f.json', 'a note'))


@pytest.mark.parametrize('bad', ['{}', '{0}', '{scene!r}', '{scene:>3}', '{a-b}', '{scene'])
def test_a_placeholder_is_a_bare_identifier(bad):
    with pytest.raises(ValueError):
        DATADIR({bad: ('x',)})


def test_validity_wants_what_is_named_and_not_what_a_pattern_matches(tmp_path):
    cls = _block({'t': DATADIR(DATAFILE('index.json'), {'{split}': DATAFILE('{scene}.npz')}, 'meta')})
    b = cls(datalake=str(tmp_path))
    root = b.path('t', ensure_dirpath=True)
    os.makedirs(os.path.join(root, 'meta'))
    assert not b.valid_topic('t')                       # index.json is named, and missing
    open(os.path.join(root, 'index.json'), 'w').close()
    assert b.valid_topic('t')                           # no split at all matches a pattern


def test_a_pattern_cannot_be_a_topic():
    with pytest.raises(TypeError, match='a pattern names many files'):
        _block({'t': DATAFILE('{scene}.npz')})
    with pytest.raises(TypeError, match='a pattern names many files'):
        _block({'g': {'t': DATAFILE('{scene}.npz')}})


# --------------------------------------------------------------------------------------------- #
# over= and VAR
# --------------------------------------------------------------------------------------------- #

def test_var_records_and_renders_and_resolves_only_when_asked():
    ref = VAR.scenery.scenes('name', n=2)[0]
    assert repr(ref) == "VAR.scenery.scenes('name', n=2)[0]"
    assert ref == copy.deepcopy(ref) == pickle.loads(pickle.dumps(ref))
    with pytest.raises(AttributeError):
        VAR._private
    with pytest.raises(TypeError, match='not iterated'):
        list(VAR.splits)
    with pytest.raises(TypeError, match='not a literal'):
        VAR.splits[object()]

    class Holder:
        var = type('V', (), {'splits': ('a', 'b')})()
    assert VAR.splits._resolve_(Holder()) == ('a', 'b')


def test_over_binds_placeholders_under_it_and_nothing_else():
    m = DATADIR({'{split}': DATAFILE('{scene}.npz')}, over={'split': ('a', 'b'), 'scene': 'its scenes'})
    assert m.over == {'split': ('a', 'b'), 'scene': 'its scenes'}
    with pytest.raises(ValueError, match="binds 'shard'"):
        DATADIR(DATAFILE('task{i}.log'), over={'shard': VAR.n_shards})
    with pytest.raises(TypeError, match='a binding is'):
        DATADIR(DATAFILE('task{i}.log'), over={'i': 3})


def test_a_var_reference_names_a_field_of_the_class():
    _block({'t': DATADIR(DATAFILE('task{i}.log'), over={'i': VAR.n_shards})}, n_shards=2)
    with pytest.raises(TypeError, match='names no field'):
        _block({'t': DATADIR(DATAFILE('task{i}.log'), over={'i': VAR.n_shardz})}, n_shards=2)


def test_topics_with_over_come_before_class_var():
    with pytest.raises(TypeError, match='before TOPICS'):
        class Late(Datablock):
            @dataclass
            class VAR(Datablock.VAR):
                splits: tuple = ('a',)
            TOPICS = {'t': DATADIR({'{split}': ('x',)}, over={'split': ('a',)})}

    class Early(Datablock):
        TOPICS = {'t': DATADIR({'{split}': ('x',)}, over={'split': VAR.splits})}

        @dataclass
        class VAR(Datablock.VAR):
            splits: tuple = ('a',)

    assert repr(Early.TOPICS['t'].over['split']) == 'VAR.splits'


# --------------------------------------------------------------------------------------------- #
# The journal, and pretty.topics
# --------------------------------------------------------------------------------------------- #

_DECLARED = {
    'config': DATADICT('config.yaml', n_scenes='int', doc='what the build resolved'),
    'scenes': DATAFRAME('scenes.parquet', split='str', scene_name='str', doc='one row per scene'),
    'sr': DATADIR({'{split}': DATADIR(DATAFILE('{scene}.npz', doc='one scene of SR'), doc='a split')},
                  over={'split': VAR.splits, 'scene': 'the scenes of the split'}, doc='the scenes, by split'),
    'logs': DATADIR,
}


def test_the_documented_rendering_reads_back_as_itself():
    text = render_topics(_DECLARED, doc=True)
    assert render_topics(literal_topics(text), doc=True) == text
    assert str(literal_topics(text)) == str(_DECLARED) == render_topics(_DECLARED)
    marker = _DECLARED['sr']
    assert pickle.loads(pickle.dumps(marker))._render_(doc=True) == marker._render_(doc=True)


class Documented(Datablock):
    TOPICS = _DECLARED

    @dataclass
    class VAR(Datablock.VAR):
        splits: tuple = ('train', 'val')

    def __build__(self):
        for topic in ('config', 'scenes'):
            open(self.path(topic, ensure_dirpath=True), 'w').close()
        for topic in ('sr', 'logs'):
            os.makedirs(self.path(topic), exist_ok=True)


def test_the_journal_keeps_the_documentation(tmp_path):
    b = Documented(datalake=str(tmp_path))
    b.build()
    entry = b.buildjournal(event='build:end', hash=b.hash, iloc=0)
    assert 'doc=' in entry['topics'] and 'VAR.splits' in entry['topics']
    assert entry.pretty.topics(doc=True) == b.pretty.topics(doc=True)
    with pytest.raises(ValueError, match='live block'):
        entry.pretty.topics(bound=True)


def test_pretty_topics_lays_out_the_declaration_in_its_order():
    p = Documented(datalake='/tmp/unused').pretty
    assert p.topics(short=True) == ['config', 'scenes', 'sr', 'logs']
    lines = p.topics().splitlines()
    assert [l.split(':')[0] for l in lines if not l.startswith(' ')] == ['config', 'scenes', 'sr', 'logs']
    assert lines[3:5] == ['  {split}/', "    DATAFILE('{scene}.npz')"]
    assert '#' not in p.topics()
    documented = p.topics(doc=True)
    assert '# the scenes, by split; split in VAR.splits; scene: the scenes of the split' in documented
    assert "split in ['train', 'val']" in p.topics(doc=True, bound=True)


# --------------------------------------------------------------------------------------------- #
# DATAFRAME
# --------------------------------------------------------------------------------------------- #

def test_a_dataframe_is_a_file_with_columns_in_order(tmp_path):
    m = DATAFRAME('s.parquet', split='str', n='int')
    assert repr(m) == "DATAFRAME('s.parquet', split='str', n='int')"
    assert m.filename == 's.parquet' and list(m.schema) == ['split', 'n']
    assert dbx.is_topicmarker(m, DATAFILE) and not dbx.is_topicmarker(m, DATADICT)
    assert _hash({'t': m}, tmp_path) != _hash({'t': DATAFRAME('s.parquet', n='int', split='str')}, tmp_path)
    with pytest.raises(TypeError):
        DATAFRAME('s.parquet', a=dict(b='int'))


def test_a_key_named_doc_takes_the_mapping_form():
    m = DATADICT('m.json', {'doc': 'str'}, doc='the docs')
    assert m.schema == {'doc': 'str'} and m.doc == 'the docs'
    assert repr(m) == "DATADICT('m.json', {'doc': 'str'})"
    assert literal_topics(render_topics({'m': m}, doc=True))['m'].schema == {'doc': 'str'}


def test_a_slice_takes_a_doc_too(tmp_path):
    from dbx.datatables import DATASLICE
    m = DATASLICE(idx='int', doc='one row per frame')
    assert repr(m) == "DATASLICE(idx='int')" and m.doc == 'one row per frame'
    assert literal_topics(render_topics({'s': m}, doc=True))['s'].doc == 'one row per frame'
