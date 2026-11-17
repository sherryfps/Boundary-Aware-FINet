"""Independently verify downloaded real industrial engineering state on CPU."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from zipfile import ZipFile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
import torch

from src.finet_adapter import extract_state_dict, state_dict_sha256
from src.industrial_training import IndustrialConfig
from src.training import make_model_optimizer, restore_checkpoint, write_json
from src.training_data import file_hash, manifest_hash


def verify(archive, extracted, local_data, output):
    archive, extracted, local_data = Path(archive), Path(extracted), Path(local_data)
    with ZipFile(archive) as zipped:
        if zipped.testzip():
            raise ValueError('Downloaded GPU recovery ZIP failed CRC.')
        members = len(zipped.namelist())
    state = extracted/'FINetIndustrial'
    manifest = json.loads((state/'split_manifest.json').read_text())
    local = json.loads((local_data/'split_manifest.json').read_text())
    for value in (manifest,local):
        if value['fingerprint'] != manifest_hash(value):
            raise ValueError('Prepared manifest fingerprint failed.')
    def semantic(value):
        value = json.loads(json.dumps(value)); value.pop('fingerprint')
        for record in value['records']:
            record.pop('mask_sha256')
        return value
    if semantic(manifest) != semantic(local):
        raise ValueError('GPU and local source identities/annotations/splits differ beyond PNG encoding.')
    masks_equal = 0
    for record in manifest['records']:
        canonical = state/'prepared_targets'/record['mask']
        if file_hash(canonical) != record['mask_sha256']:
            raise ValueError('Downloaded canonical target differs from GPU manifest.')
        with Image.open(canonical) as gpu_mask, Image.open(local_data/record['mask']) as cpu_mask:
            if not np.array_equal(np.asarray(gpu_mask.convert('L')),np.asarray(cpu_mask.convert('L'))):
                raise ValueError(f'Local/GPU target pixels differ: {record["id"]}')
        masks_equal += 1
    torch.set_num_threads(2)
    shared = torch.load(state/'shared_initialization.pth',map_location='cpu',weights_only=True)
    initial_digest = state_dict_sha256(extract_state_dict(shared))
    variants = {}
    for variant in ('baseline','boundary'):
        folder = state/'smoke'/variant
        saved_report = json.loads((folder/'smoke_report.json').read_text())
        checkpoint = folder/'smoke_checkpoint.pth'
        payload = torch.load(checkpoint,map_location='cpu',weights_only=True)
        config = IndustrialConfig(**{**payload['config'],'device':'cpu'})
        model,optimizer,scheduler = make_model_optimizer(config,torch.device('cpu'),load_path=checkpoint)
        generator = torch.Generator().manual_seed(config.seed)
        restored = restore_checkpoint(checkpoint,model,optimizer,scheduler,generator,config,manifest,allow_smoke=True)
        for key,value in model.state_dict().items():
            torch.testing.assert_close(value,payload['model_state_dict'][key],rtol=0,atol=0)
        for key,values in payload['optimizer_state_dict']['state'].items():
            for name,value in values.items():
                torch.testing.assert_close(optimizer.state_dict()['state'][key][name],value,rtol=0,atol=0)
                if not torch.isfinite(value).all():
                    raise ValueError('Nonfinite GPU optimizer state.')
        if scheduler.state_dict() != payload['scheduler_state_dict']:
            raise ValueError('Downloaded scheduler differs after restoration.')
        if not torch.equal(generator.get_state(),payload['rng_state']['loader']):
            raise ValueError('Downloaded loader RNG differs after restoration.')
        if restored['initial_model_sha256'] != initial_digest or saved_report['initial_model_sha256'] != initial_digest:
            raise ValueError('GPU variants do not share the complete initialization.')
        variants[variant] = {'training_status':payload['training_status'],
                             'completed_epochs':payload['completed_epochs'],'global_step':payload['global_step'],
                             'checkpoint_sha256':file_hash(checkpoint),'model_tensors':len(model.state_dict()),
                             'optimizer_states':len(optimizer.state_dict()['state']),
                             'scheduler_count':scheduler.cnt,'model_optimizer_scheduler_loader_restored_exactly':True,
                             'saved_cuda_rng_states':len(payload['rng_state']['cuda']),
                             'gpu_smoke_losses':[r['loss'] for r in saved_report['steps']],
                             'resumed_gpu_smoke_loss':saved_report['resumed_optimizer_step']['loss'],
                             'peak_allocated_bytes':saved_report['hardware']['peak_allocated_bytes']}
        del model,optimizer,scheduler
    result = {'status':'passed','scope':'real GPU engineering backup; zero completed industrial training epochs',
              'archive_bytes':archive.stat().st_size,'archive_sha256':file_hash(archive),'zip_members':members,
              'gpu_split_fingerprint':manifest['fingerprint'],'local_split_fingerprint':local['fingerprint'],
              'source_identities_and_splits_identical':True,'canonical_masks_pixel_identical':masks_equal,
              'png_encoding_note':'PNG byte hashes can differ across Pillow/zlib versions; all canonical target pixels were compared exactly.',
              'initial_model_sha256':initial_digest,'variants':variants}
    write_json(Path(output),result)
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',default='outputs/industrial_setup/gpu_smoke_recovery.zip')
    parser.add_argument('--extracted',default='outputs/industrial_setup/gpu_smoke_version001')
    parser.add_argument('--local-data',default='data/MVTecAD_supervised_v1')
    parser.add_argument('--output',default='outputs/industrial_setup/gpu_smoke_backup_verification.json')
    args=parser.parse_args()
    verify(args.archive,args.extracted,args.local_data,args.output)
