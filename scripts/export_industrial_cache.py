"""Run interactively on Kaggle CPU, then Quick Save WITH outputs; no training."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
from time import perf_counter


def export():
    import kagglehub
    working = Path('/kaggle/working')
    if not working.is_dir():
        raise RuntimeError('Run this cache-copy cell inside an interactive Kaggle CPU notebook.')
    started = perf_counter()
    handle = 'mahesh2179/finet-industrial-defect-segmentation/versions/1'
    print(json.dumps({'event':'cache_source_requested', 'handle':handle}), flush=True)
    snapshot = Path(kagglehub.notebook_output_download(handle))
    candidates = list(snapshot.rglob('MVTecAD_supervised_v1/split_manifest.json'))
    if snapshot.name == 'MVTecAD_supervised_v1' and (snapshot/'split_manifest.json').is_file():
        candidates.append(snapshot/'split_manifest.json')
    candidates = sorted(set(candidates))
    if len(candidates) != 1:
        raise ValueError('The requested immutable Version-1 output does not expose exactly one prepared cache.')
    source = candidates[0].parent.resolve()
    manifest_bytes = candidates[0].read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest['fingerprint'] != '219361ad92613f34bb36fe9beee42998dfe03f49aa436aedc0c168bafb0980fa':
        raise ValueError('Refusing a different industrial split/cache.')
    if len(manifest['records']) != 5354 or [len(manifest[k]) for k in ('train_ids','validation_ids','test_ids')] != [3728,813,813]:
        raise ValueError('Prepared cache counts differ from the registered experiment.')
    target = (working/'MVTecAD_supervised_v1').resolve()
    target.mkdir(parents=True, exist_ok=True)

    def digest(path):
        with path.open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()

    def copy_verified(job):
        relative, expected = job
        origin, destination = (source/relative).resolve(), (target/relative).resolve()
        if not origin.is_relative_to(source) or not destination.is_relative_to(target):
            raise ValueError('Unsafe prepared-cache path.')
        if not destination.is_file() or digest(destination) != expected:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name+'.copying')
            shutil.copyfile(origin, temporary)
            if digest(temporary) != expected:
                raise ValueError('Prepared file SHA mismatch: '+relative)
            temporary.replace(destination)
        return destination.stat().st_size

    jobs = [(row[k],row[k+'_sha256']) for row in manifest['records'] for k in ('image','mask')]
    print(json.dumps({'event':'cache_copy_started', 'source':str(source), 'files':len(jobs),
                      'split_fingerprint':manifest['fingerprint']}), flush=True)
    byte_count = 0
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(copy_verified, job) for job in jobs]
        for count, future in enumerate(as_completed(futures), 1):
            byte_count += future.result()
            if count % 1000 == 0 or count == len(jobs):
                print(json.dumps({'event':'cache_copy_progress', 'verified_files':count,
                                  'total_files':len(jobs), 'seconds':perf_counter()-started}), flush=True)
    (target/'split_manifest.json').write_bytes(manifest_bytes)
    report = {'event':'prepared_cache_export_complete', 'source_handle':handle,
              'source':str(source), 'target':str(target), 'verified_image_mask_files':len(jobs),
              'image_count':5354, 'verified_bytes':byte_count,
              'split_fingerprint':manifest['fingerprint'], 'seconds':perf_counter()-started,
              'training_epochs_executed':0, 'gpu_required':False,
              'next_action':'Quick Save with Save output for this version selected; verify the saved cache output.'}
    (working/'prepared_cache_export.json').write_text(json.dumps(report,indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)
    return report


if __name__ == '__main__':
    export()
