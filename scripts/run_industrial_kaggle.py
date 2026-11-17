"""Bounded, recoverable industrial A/B supervisor. Never touches COD checkpoints."""

import argparse
from dataclasses import asdict
from functools import lru_cache
import io
import json
import math
from pathlib import Path
import shutil
import sys
from time import monotonic
from zipfile import ZipFile, ZIP_DEFLATED

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from src.finet_adapter import extract_state_dict, state_dict_sha256
from src.industrial_data import DOMAIN, CATEGORIES, load_industrial_manifest, prepare_supervised
from src.industrial_training import IndustrialConfig, run_industrial_smoke, run_industrial_training
from src.training import write_json
from src.training_data import file_hash

ARCHIVE_SHA256 = 'cf4313b13603bec67abb49ca959488f7eedce2a9f7795ec54446c649ac98cd3d'


@lru_cache(maxsize=1)
def expected_state_layout():
    from Model.FINet import FINet
    model = FINet(backbone='efficientb0', channels=(8,24,32,64), pretrained_backbone=False)
    return {name:(tuple(value.shape),value.dtype) for name,value in model.state_dict().items()}


def checkpoint_rank(payload):
    if (payload.get('application_domain') != DOMAIN or payload.get('dataset_protocol') != 'custom_supervised_mvtec_v1'
            or payload.get('training_status') not in ('baseline_training', 'boundary_training')
            or not isinstance(payload.get('optimizer_state_dict'), dict)
            or not isinstance(payload.get('scheduler_state_dict'), dict) or not payload.get('rng_state')):
        raise ValueError('Recovery is not a complete industrial training state.')
    completed = payload.get('completed_epochs')
    if type(completed) is not int or not 1 <= completed <= payload['config']['epochs']:
        raise ValueError('Recovery epoch is invalid.')
    if (payload['scheduler_state_dict'].get('cnt') != completed
            or len(payload['optimizer_state_dict'].get('state', {})) != 581
            or not {'python','numpy','torch','cuda','loader'} <= payload['rng_state'].keys()):
        raise ValueError('Recovery optimizer, scheduler or RNG state is incomplete.')
    layout = {name:(tuple(value.shape),value.dtype) for name,value in extract_state_dict(payload).items()}
    if layout != expected_state_layout():
        raise ValueError('Recovery lacks the full 965-tensor FINet state.')
    settings = payload['config']
    scheduler = payload['scheduler_state_dict']
    expected_lr = ((settings['learning_rate']-settings['min_lr'])/2
                   * math.cos(math.pi/(settings['epochs']-1)*completed)
                   + (settings['learning_rate']+settings['min_lr'])/2) if settings['epochs'] > 1 else settings['learning_rate']
    if (scheduler.get('max_epoch') != settings['epochs'] or scheduler.get('max_lr') != settings['learning_rate']
            or scheduler.get('min_lr') != settings['min_lr']
            or not math.isclose(scheduler.get('current_lr',float('nan')),expected_lr,rel_tol=1e-10,abs_tol=1e-12)):
        raise ValueError('Recovery learning-rate schedule is inconsistent with its completed epoch.')
    for state in payload['optimizer_state_dict']['state'].values():
        if any(isinstance(value, torch.Tensor) and not torch.isfinite(value).all() for value in state.values()):
            raise ValueError('Recovery has nonfinite optimizer state.')
        if float(state.get('step', -1)) != payload.get('global_step'):
            raise ValueError('Recovery optimizer update count differs from global step.')
    return completed


def saved_initialization_identity(directory):
    """Recognize an engineering snapshot without crediting any trained epoch."""
    directory = Path(directory)
    shared = directory/'shared_initialization.pth'
    reports = [directory/'smoke'/v/'smoke_report.json' for v in ('baseline','boundary')]
    if not shared.is_file() or not all(path.is_file() for path in reports):
        return None
    payload = torch.load(shared,map_location='cpu',weights_only=True)
    if payload.get('training_status') != 'initialization' or payload.get('application_domain') != DOMAIN:
        raise ValueError('Saved industrial engineering initialization has invalid provenance.')
    state = extract_state_dict(payload)
    if {name:(tuple(value.shape),value.dtype) for name,value in state.items()} != expected_state_layout():
        raise ValueError('Saved initialization is not the complete industrial FINet model.')
    digest = state_dict_sha256(state)
    for variant, path in zip(('baseline','boundary'),reports):
        report = json.loads(path.read_text())
        initialization = directory/'smoke'/variant/'initialization.pth'
        if (report.get('status') != 'passed' or report.get('initial_model_sha256') != digest
                or report.get('split_fingerprint') != payload.get('split_fingerprint')
                or not report.get('hardware',{}).get('device','').startswith('cuda')
                or not initialization.is_file() or file_hash(initialization) != report.get('initialization_sha256')):
            raise ValueError('Saved industrial GPU smoke evidence does not match its initial weights.')
    return digest


def restore_latest(inputs, state):
    """Choose highest completed A/B tuple, never silently roll back an existing run."""
    state = Path(state)
    ranks = []
    for archive in sorted(Path(inputs).rglob('industrial_recovery.zip')):
        with ZipFile(archive) as zipped:
            if sum(m.file_size for m in zipped.infolist()) > 3*1024**3:
                raise ValueError('Recovery archive unexpectedly large.')
            scores, identities = [], []
            for variant in ('baseline', 'boundary'):
                name = f'FINetIndustrial/experiments/{variant}/last.pth'
                if name in zipped.namelist():
                    payload = torch.load(io.BytesIO(zipped.read(name)), map_location='cpu', weights_only=True)
                    scores.append(checkpoint_rank(payload))
                    identities.append((variant, state_dict_sha256(extract_state_dict(payload))))
                else:
                    scores.append(0)
            if scores != [0,0]:
                ranks.append((tuple(scores), tuple(identities), archive))
    # Kaggle may expand ZIP inputs. Saved notebook outputs also expose their
    # FINetIndustrial directory directly; do not require a surviving nested ZIP.
    initializations = []
    for directory in sorted(Path(inputs).rglob('FINetIndustrial')):
        if not directory.is_dir():
            continue
        identity = saved_initialization_identity(directory)
        if identity:
            initializations.append((identity,directory))
        if not (directory/'experiments').is_dir():
            continue
        scores, identities = [], []
        for variant in ('baseline','boundary'):
            path = directory/'experiments'/variant/'last.pth'
            if path.is_file():
                payload = torch.load(path,map_location='cpu',weights_only=True)
                scores.append(checkpoint_rank(payload))
                identities.append((variant,state_dict_sha256(extract_state_dict(payload))))
            else:
                scores.append(0)
        if scores != [0,0]:
            ranks.append((tuple(scores),tuple(identities),directory))
    own_scores = []
    own_identities = []
    for variant in ('baseline', 'boundary'):
        path = state/'experiments'/variant/'last.pth'
        if path.exists():
            payload = torch.load(path,map_location='cpu',weights_only=True)
            own_scores.append(checkpoint_rank(payload))
            own_identities.append((variant,state_dict_sha256(extract_state_dict(payload))))
        else:
            own_scores.append(0)
    if ranks and tuple(own_scores) == max(r[0] for r in ranks):
        if any(r[1] != tuple(own_identities) for r in ranks if r[0] == tuple(own_scores)):
            raise ValueError('Working and attached industrial states conflict at the same completed epoch.')
    if not ranks and not any(own_scores) and initializations and (not state.exists() or not any(state.iterdir())):
        if len({item[0] for item in initializations}) != 1:
            raise ValueError('Attached engineering snapshots contain different industrial initial weights.')
        source = initializations[0][1]
        shutil.copytree(source,state,dirs_exist_ok=True)
        return {'source':str(source),'completed':[0,0],'storage':'verified GPU engineering snapshot',
                'initial_model_sha256':initializations[0][0]}
    if not ranks or tuple(own_scores) >= max(r[0] for r in ranks):
        return {'source': 'current working state', 'completed': own_scores}
    best_score = max(r[0] for r in ranks)
    tied = [r for r in ranks if r[0] == best_score]
    if len({r[1] for r in tied}) != 1:
        raise ValueError('Conflicting industrial checkpoints at the same completed epochs; review required.')
    archive = tied[0][2]
    if state.exists() and any(state.iterdir()):
        raise ValueError('Attached recovery is newer than nonempty working state; preserve both and review before replacing.')
    state.mkdir(parents=True, exist_ok=True)
    if archive.is_dir():
        shutil.copytree(archive,state,dirs_exist_ok=True)
        return {'source':str(archive),'completed':list(best_score),'storage':'expanded notebook output'}
    with ZipFile(archive) as zipped:
        if zipped.testzip():
            raise ValueError('Industrial recovery ZIP CRC failed.')
        for member in zipped.infolist():
            if not member.filename.startswith('FINetIndustrial/'):
                raise ValueError('Unexpected recovery archive root.')
            relative = Path(member.filename).relative_to('FINetIndustrial')
            destination = (state/relative).resolve()
            if not destination.is_relative_to(state.resolve()):
                raise ValueError('Unsafe recovery path.')
            if not member.is_dir():
                destination.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(member) as source, destination.open('wb') as target:
                    shutil.copyfileobj(source,target)
    return {'source': str(archive), 'completed': list(best_score)}


def find_original(inputs, temporary):
    candidates = [path.parent.parent.parent for path in Path(inputs).rglob('bottle/train/good')]
    candidates = [p for p in candidates if all((p/category/'train/good').is_dir() for category in CATEGORIES)]
    if len(candidates) == 1:
        return candidates[0]
    archives = list(Path(inputs).rglob('mvtec_anomaly_detection.tar.xz'))
    if len(archives) != 1:
        raise ValueError('Attach exactly one original MVTec AD archive or extracted 15-category input.')
    if file_hash(archives[0]) != ARCHIVE_SHA256:
        raise ValueError('Kaggle archive SHA differs from the fully verified supplied archive.')
    from scripts.download_mvtec import verify_and_extract
    original = Path(temporary)/'original'
    verify_and_extract(archives[0],original)
    return original


def recovery_zip(state, target):
    state, target = Path(state), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.zip.tmp')
    latest_preview = {}
    for variant in ('baseline','boundary'):
        folder = state/'experiments'/variant/'validation_samples'
        epochs = sorted(p for p in folder.glob('epoch_*') if p.is_dir())
        if epochs:
            latest_preview[variant] = epochs[-1].name
    with ZipFile(temporary,'w',ZIP_DEFLATED,compresslevel=1) as zipped:
        for path in sorted(state.rglob('*')):
            relative = path.relative_to(state)
            if relative.as_posix() == 'recovery_archive.json':
                # The archive report describes the completed ZIP and must not
                # recursively change that ZIP when the finalizer runs again.
                continue
            if 'validation_samples' in relative.parts:
                # All historical previews stay in the saved version's outputs.
                # Recovery carries the newest preview plus every checkpoint/log,
                # avoiding multi-GB raw .npy history in the portable resume ZIP.
                if len(relative.parts) < 5 or relative.parts[3] != latest_preview.get(relative.parts[1]):
                    continue
            if path.is_file() and not path.name.endswith('.tmp'):
                zipped.write(path, 'FINetIndustrial/'+path.relative_to(state).as_posix())
    with ZipFile(temporary) as zipped:
        if zipped.testzip():
            raise ValueError('Generated recovery archive failed integrity check.')
    temporary.replace(target)
    return {'path': str(target), 'sha256': file_hash(target), 'bytes': target.stat().st_size}


def run(inputs='/kaggle/input', state='/kaggle/working/FINetIndustrial', temporary='/kaggle/temp/finet_industrial',
        train=False, max_hours=7.5, max_new_epochs=None):
    started = monotonic()
    if not torch.cuda.is_available():
        raise ValueError('Kaggle CUDA GPU is not available. No long training was launched.')
    if not 0 < max_hours <= 8:
        raise ValueError('Bound each saved session to at most 8 hours.')
    state = Path(state)
    recovery = restore_latest(inputs,state)
    print(json.dumps({'event':'industrial_recovery_selected',**recovery}),flush=True)
    caches = sorted(path.parent for path in Path(inputs).rglob('MVTecAD_supervised_v1/split_manifest.json'))
    data = state.parent/'MVTecAD_supervised_v1'
    if caches:
        manifests = [load_industrial_manifest(path/'split_manifest.json',path) for path in caches]
        if len({m['fingerprint'] for m in manifests}) != 1:
            raise ValueError('Attached prepared industrial caches disagree; do not silently change data.')
        data, manifest = caches[0], manifests[0]
    elif (data/'split_manifest.json').is_file():
        manifest = load_industrial_manifest(data/'split_manifest.json',data)
    else:
        original = find_original(inputs,temporary)
        manifest = prepare_supervised(original,data)
    if (manifest['seed'] != 123456 or set(manifest['categories']) != set(CATEGORIES)
            or manifest['source_counts'] != {'MVTec AD':5354}):
        raise ValueError('This registered run requires the complete 15-category industrial protocol.')
    state.mkdir(parents=True,exist_ok=True)
    previous_manifest = state/'split_manifest.json'
    canonical_masks = state/'prepared_targets'
    if previous_manifest.exists():
        saved = json.loads(previous_manifest.read_text())
        # Pillow/zlib releases may encode identical binary PNGs differently.
        # Preserve the original prepared target bytes instead of changing a split
        # fingerprint on resume. All source identities/labels/partitions must match.
        def normalized(value):
            value = json.loads(json.dumps(value))
            value.pop('fingerprint',None)
            for row in value['records']:
                row.pop('mask_sha256',None)
            return value
        if normalized(saved) != normalized(manifest):
            raise ValueError('Industrial source identities or custom split changed; recovery refused.')
        if saved['fingerprint'] != manifest['fingerprint']:
            for row in saved['records']:
                source = canonical_masks/row['mask']
                if file_hash(source) != row['mask_sha256']:
                    raise ValueError('Canonical prepared target is missing or changed.')
                shutil.copyfile(source,data/row['mask'])
            write_json(data/'split_manifest.json',saved)
            manifest = load_industrial_manifest(data/'split_manifest.json',data)
    else:
        for row in manifest['records']:
            target = canonical_masks/row['mask']
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(data/row['mask'],target)
    write_json(state/'split_manifest.json',manifest)
    write_json(state/'data_location.json',{'data_root':str(data),'split_fingerprint':manifest['fingerprint']})
    print(json.dumps({'event':'industrial_data_verified','fingerprint':manifest['fingerprint'],
                      'counts':{key:len(manifest[key]) for key in ('train_ids','validation_ids','test_ids')},
                      'categories':manifest['categories']}),flush=True)
    reports = {}
    shared = state/'shared_initialization.pth'
    for variant in ('baseline','boundary'):
        config = IndustrialConfig(**json.loads((Path(__file__).resolve().parents[1]/'configs'/f'industrial_{variant}.json').read_text()))
        config.data_root, config.split_manifest = str(data), str(data/'split_manifest.json')
        config.device = 'cuda:0'
        config.output_dir = str(state/'smoke'/variant)
        config.initial_checkpoint = str(shared) if shared.exists() else None
        if variant == 'boundary' and not shared.exists():
            raise ValueError('Boundary experiment must start from the original industrial full initialization.')
        report_path = Path(config.output_dir)/'smoke_report.json'
        if report_path.exists():
            report = json.loads(report_path.read_text())
        else:
            report = run_industrial_smoke(config,steps=3)
        if variant == 'baseline' and not shared.exists():
            shutil.copyfile(Path(config.output_dir)/'initialization.pth',shared)
        config.initial_checkpoint = str(Path(config.output_dir)/'initialization.pth')
        config.output_dir = str(state/'experiments'/variant)
        if state_dict_sha256(extract_state_dict(torch.load(shared,map_location='cpu',weights_only=True))) != report['initial_model_sha256']:
            raise ValueError('Industrial A/B initialization does not match exactly.')
        print(json.dumps({'event':'industrial_smoke_verified','variant':variant,'gpu':report['hardware'].get('gpu'),
                          'losses':[r['loss'] for r in report['steps']],
                          'steady_step_seconds':report['measured_steady_step_seconds'],
                          'steps_per_epoch':report['training_steps_per_epoch'],
                          'estimated_training_only_hours':report['measured_steady_step_seconds']*report['training_steps_per_epoch']*config.epochs/3600,
                          'estimate_excludes':'validation, startup, checkpoint I/O and quota waits'}),flush=True)
        reports[variant] = report
        if train:
            remaining = max_hours - (monotonic()-started)/3600
            if remaining < .2:
                break
            checkpoint = Path(config.output_dir)/'last.pth'
            result = run_industrial_training(config, report_path, str(checkpoint) if checkpoint.exists() else None,
                                             remaining,max_new_epochs)
            print(json.dumps({'event':'industrial_stage_result','variant':variant,**result}),flush=True)
            if not result['training_complete']:
                break
    proof = recovery_zip(state,state.parent/'industrial_recovery.zip')
    write_json(state/'recovery_archive.json',proof)
    return {'training_requested':train,'recovery':proof,'smoke_variants_verified':list(reports)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',action='store_true')
    parser.add_argument('--max-hours',type=float,default=7.5)
    parser.add_argument('--max-new-epochs',type=int)
    args = parser.parse_args()
    try:
        print(json.dumps(run(train=args.train,max_hours=args.max_hours,max_new_epochs=args.max_new_epochs),indent=2),flush=True)
    finally:
        state = Path('/kaggle/working/FINetIndustrial')
        if state.exists():
            proof = recovery_zip(state,state.parent/'industrial_recovery.zip')
            write_json(state/'recovery_archive.json',proof)
            print(json.dumps({'event':'industrial_recovery_saved',**proof}),flush=True)
