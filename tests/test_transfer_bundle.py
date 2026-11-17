import json

import pytest

from scripts.verify_transfer_bundle import digest_file, verify_bundle


def bundle(tmp_path):
    entries = []
    for name, value in [('project/first.txt', b'code'), ('project/last.pth', b'trained state')]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        entries.append({'path': name, 'bytes': len(value), 'sha256': digest_file(path)})
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(entry) + '\n' for entry in entries))
    completion = {'status': 'passed', 'manifest_sha256': digest_file(manifest), 'files': 2,
                  'bytes': sum(entry['bytes'] for entry in entries)}
    (tmp_path / 'TRANSFER_COMPLETE.json').write_text(json.dumps(completion))
    return entries


def test_verifies_all_files_and_bytes(tmp_path):
    entries = bundle(tmp_path)
    result = verify_bundle(tmp_path, workers=2)
    assert result['status'] == 'passed' and result['files'] == len(entries)
    assert result['bytes'] == sum(entry['bytes'] for entry in entries)


def test_rejects_corruption_in_last_file(tmp_path):
    bundle(tmp_path)
    (tmp_path / 'project/last.pth').write_bytes(b'broken! state')
    with pytest.raises(ValueError, match='Missing or changed file'):
        verify_bundle(tmp_path, workers=2)


def test_rejects_unfinished_copy(tmp_path):
    (tmp_path / 'manifest.jsonl').write_text('')
    with pytest.raises(ValueError, match='unfinished copy'):
        verify_bundle(tmp_path)


def test_rejects_path_escape_even_with_matching_manifest_digest(tmp_path):
    entries = bundle(tmp_path)
    entries[0]['path'] = '../outside.txt'
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(entry) + '\n' for entry in entries))
    completion = json.loads((tmp_path / 'TRANSFER_COMPLETE.json').read_text())
    completion['manifest_sha256'] = digest_file(manifest)
    (tmp_path / 'TRANSFER_COMPLETE.json').write_text(json.dumps(completion))
    with pytest.raises(ValueError, match='Unsafe manifest path'):
        verify_bundle(tmp_path, workers=2)
