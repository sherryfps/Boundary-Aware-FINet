"""Download the user-supplied original MVTec AD archive, with resumable transfer."""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import time
import urllib.request

URL = ('https://www.mydrive.ch/shares/163178/c8212c2166a9e4dfe22dd1bed476a345/'
       'download/420938113-1629960298/mvtec_anomaly_detection.tar.xz')
EXPECTED_BYTES = 5264982680


def download(destination, url=URL):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.part')
    if destination.exists():
        if destination.stat().st_size != EXPECTED_BYTES:
            raise ValueError('Existing archive has the wrong size; preserve it and choose another destination.')
        return destination
    offset = temporary.stat().st_size if temporary.exists() else 0
    request = urllib.request.Request(url, headers={'Range': f'bytes={offset}-'} if offset else {})
    with urllib.request.urlopen(request, timeout=60) as response:
        if offset and (response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-')):
            raise ValueError('Server did not honor the resume range; partial file preserved.')
        expected = offset + int(response.headers['Content-Length'])
        if expected != EXPECTED_BYTES:
            raise ValueError(f'Unexpected archive length: {expected}')
        last = 0
        with temporary.open('ab' if offset else 'wb') as output:
            while chunk := response.read(4 * 1024 * 1024):
                output.write(chunk)
                offset += len(chunk)
                if time.monotonic() - last >= 30:
                    print(json.dumps({'event': 'download', 'bytes': offset, 'total_bytes': expected}), flush=True)
                    last = time.monotonic()
    if temporary.stat().st_size != expected:
        raise ValueError('Truncated download; partial file retained for resuming.')
    temporary.replace(destination)
    return destination


def verify_and_extract(archive, destination):
    archive, destination = Path(archive), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with archive.open('rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    # Reading the entire xz stream verifies its built-in integrity checks. Reject
    # links and traversal rather than trusting names from an external archive.
    members = 0
    with tarfile.open(archive, 'r:xz') as source:
        for member in source:
            resolved = (destination / member.name).resolve()
            if not resolved.is_relative_to(destination.resolve()) or not (member.isfile() or member.isdir()):
                raise ValueError(f'Unsafe archive member: {member.name}')
            source.extract(member, destination, filter='data')
            members += 1
    report = {'url': URL, 'archive_bytes': archive.stat().st_size,
              'archive_sha256': digest.hexdigest(), 'tar_members': members,
              'integrity': 'Full xz/tar decode; no publisher SHA256 supplied for independent authenticity comparison.',
              'license': 'CC BY-NC-SA 4.0; academic non-commercial use only.'}
    (destination / 'archive_verification.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=Path('data/downloads/mvtec_anomaly_detection.tar.xz'))
    parser.add_argument('--extract-to', type=Path, default=Path('data/MVTecAD_original'))
    args = parser.parse_args()
    verify_and_extract(download(args.archive), args.extract_to)


if __name__ == '__main__':
    main()
