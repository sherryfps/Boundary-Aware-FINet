"""Recovery guards and notebook checks; dictionary fixtures are synthetic unit data."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from scripts import run_industrial_kaggle as runner
from src.industrial_data import DOMAIN


def payload_fixture():
    return {'application_domain':DOMAIN,'dataset_protocol':'custom_supervised_mvtec_v1',
            'training_status':'baseline_training','completed_epochs':1,'global_step':10,
            'config':{'epochs':2,'learning_rate':.00026,'min_lr':1e-7},
            'model_state_dict':{'mock_tensor':torch.tensor(0.)},
            'optimizer_state_dict':{'state':{i:{'step':torch.tensor(10.),'exp_avg':torch.tensor(0.)} for i in range(581)}},
            'scheduler_state_dict':{'cnt':1,'current_lr':1e-7,'max_epoch':2,'max_lr':.00026,'min_lr':1e-7},
            'rng_state':{key:[] for key in ('python','numpy','torch','cuda','loader')}}


def test_resume_rejects_lost_optimizer_schedule_or_rng(monkeypatch):
    monkeypatch.setattr(runner,'expected_state_layout',lambda:{'mock_tensor':((),torch.float32)})
    # The mock layout is intentionally a scalar fixture, never a model checkpoint.
    payload = payload_fixture()
    assert runner.checkpoint_rank(payload) == 1
    for mutate in (lambda p:p['optimizer_state_dict']['state'].pop(0),
                   lambda p:p['scheduler_state_dict'].update(cnt=0),
                   lambda p:p['scheduler_state_dict'].update(current_lr=.5),
                   lambda p:p['rng_state'].pop('loader'),
                   lambda p:p['optimizer_state_dict']['state'][0].update(step=torch.tensor(9.))):
        broken = deepcopy(payload)
        mutate(broken)
        with pytest.raises(ValueError,match='Recovery'):
            runner.checkpoint_rank(broken)


def test_resume_rejects_camouflage_or_incomplete_model(monkeypatch):
    monkeypatch.setattr(runner,'expected_state_layout',lambda:{'other_tensor':((),torch.float32)})
    with pytest.raises(ValueError,match='965'):
        runner.checkpoint_rank(payload_fixture())
    payload=payload_fixture(); payload['application_domain']='Camouflaged Objects'
    with pytest.raises(ValueError,match='industrial'):
        runner.checkpoint_rank(payload)


def test_portable_notebook_compiles_and_defaults_to_smoke_only():
    root=Path(__file__).resolve().parents[1]
    notebook=json.loads((root/'notebooks/industrial_training_kaggle.ipynb').read_text())
    code=[]
    for index,cell in enumerate(notebook['cells']):
        if cell['cell_type']=='code':
            source=''.join(cell['source'])
            compile(source,f'industrial_notebook_cell_{index}','exec')
            code.append(source)
    assert 'RUN_LONG_TRAINING = False' in '\n'.join(code)
    assert all('timm.create_model' not in source for source in code)


def test_recovery_reads_expanded_notebook_output_and_refuses_conflict(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'expected_state_layout',lambda:{'mock_tensor':((),torch.float32)})
    source=tmp_path/'inputs'/'notebook_output'/'FINetIndustrial'
    checkpoint=source/'experiments'/'baseline'/'last.pth'
    checkpoint.parent.mkdir(parents=True)
    torch.save(payload_fixture(),checkpoint)
    state=tmp_path/'working'/'FINetIndustrial'
    result=runner.restore_latest(tmp_path/'inputs',state)
    assert result['completed']==[1,0]
    assert result['storage']=='expanded notebook output'
    assert (state/'experiments'/'baseline'/'last.pth').is_file()
    broken=payload_fixture(); broken['model_state_dict']['mock_tensor']=torch.tensor(1.)
    torch.save(broken,state/'experiments'/'baseline'/'last.pth')
    with pytest.raises(ValueError,match='conflict'):
        runner.restore_latest(tmp_path/'inputs',state)


def test_portable_recovery_keeps_latest_preview_all_logs_and_targets(tmp_path):
    from zipfile import ZipFile
    state=tmp_path/'FINetIndustrial'
    for name in ('experiments/baseline/validation_samples/epoch_001/image/probability.npy',
                 'experiments/baseline/validation_samples/epoch_002/image/probability.npy',
                 'experiments/baseline/epochs.jsonl','prepared_targets/GT/example.png'):
        path=state/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(b'explicit synthetic unit fixture')
    archive=tmp_path/'industrial_recovery.zip'
    runner.recovery_zip(state,archive)
    with ZipFile(archive) as zipped:
        names=zipped.namelist()
        assert not any('epoch_001' in name for name in names)
        assert any('epoch_002' in name for name in names)
        assert 'FINetIndustrial/experiments/baseline/epochs.jsonl' in names
        assert 'FINetIndustrial/prepared_targets/GT/example.png' in names


def test_saved_gpu_engineering_state_preserves_initialization_without_trained_epochs(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'expected_state_layout',lambda:{'mock_tensor':((),torch.float32)})
    source=tmp_path/'inputs'/'snapshot'/'FINetIndustrial'
    source.mkdir(parents=True)
    payload=payload_fixture(); payload.update(training_status='initialization',completed_epochs=0,global_step=0,split_fingerprint='synthetic-unit-split')
    torch.save(payload,source/'shared_initialization.pth')
    digest=runner.state_dict_sha256(payload['model_state_dict'])
    for variant in ('baseline','boundary'):
        folder=source/'smoke'/variant; folder.mkdir(parents=True)
        torch.save(payload,folder/'initialization.pth')
        (folder/'smoke_report.json').write_text(json.dumps({'status':'passed','initial_model_sha256':digest,
             'split_fingerprint':'synthetic-unit-split','hardware':{'device':'cuda:0'},
             'initialization_sha256':runner.file_hash(folder/'initialization.pth')}))
    state=tmp_path/'working'/'FINetIndustrial'
    result=runner.restore_latest(tmp_path/'inputs',state)
    assert result['completed']==[0,0]
    assert result['initial_model_sha256']==digest
    assert (state/'shared_initialization.pth').is_file()
    assert not (state/'experiments'/'baseline'/'last.pth').exists()


def test_recovery_report_does_not_change_the_archive_it_describes(tmp_path):
    state=tmp_path/'FINetIndustrial'; state.mkdir()
    (state/'epochs.jsonl').write_text('explicit synthetic fixture\n')
    archive=tmp_path/'industrial_recovery.zip'
    first=runner.recovery_zip(state,archive)
    (state/'recovery_archive.json').write_text(json.dumps(first))
    second=runner.recovery_zip(state,archive)
    assert second['sha256']==first['sha256']
    assert second['bytes']==first['bytes']
