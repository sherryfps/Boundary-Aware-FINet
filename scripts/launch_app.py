"""Start the full local Gradio app once and open it after it is ready."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parents[1]


def app_ready(url):
    try:
        with urllib.request.urlopen(url + '/config', timeout=1) as response:
            return json.load(response).get('title') == 'FINet Vision AI'
    except (OSError, ValueError, urllib.error.URLError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--no-browser', action='store_true', help='Start/check the app without opening a browser.')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('Port must be between 1 and 65535.')
    url = f'http://127.0.0.1:{args.port}'
    logs = ROOT / 'outputs/app_logs'
    logs.mkdir(parents=True, exist_ok=True)

    # A second double-click must not start another server during initialization.
    import msvcrt
    with (logs / f'launcher_{args.port}.lock').open('a+b') as lock:
        lock.seek(0, 2)
        if not lock.tell():
            lock.write(b'\0')
            lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            print('FINet is already starting. Its browser window will open when ready.')
            return 0
        try:
            if app_ready(url):
                print(f'FINet is already running: {url}')
            else:
                with socket.socket() as probe:
                    try:
                        probe.bind(('127.0.0.1', args.port))
                    except OSError:
                        print(f'Port {args.port} is in use by another process. Close that process or use --port NUMBER.')
                        return 1
                for variant in ('baseline', 'boundary'):
                    if not (ROOT / f'checkpoints/{variant}/last.pth').is_file():
                        print(f'Missing final {variant} checkpoint. Restore the project checkpoints folder first.')
                        return 1
                environment = os.environ.copy()
                environment['TORCH_HOME'] = str(ROOT / 'model_cache/torch')
                environment['GRADIO_ANALYTICS_ENABLED'] = 'False'
                log_path = logs / f'server_{args.port}.log'
                with log_path.open('a', encoding='utf-8') as log:
                    log.write(f'\n--- FINet start {datetime.now().isoformat(timespec="seconds")} ---\n')
                    log.flush()
                    process = subprocess.Popen(
                        [sys.executable, '-u', str(ROOT / 'app.py'), '--device', 'cpu', '--port', str(args.port)],
                        cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW,
                    )
                (logs / f'server_{args.port}.json').write_text(
                    json.dumps({'pid': process.pid, 'url': url, 'project_root': str(ROOT),
                                'log': str(log_path)}, indent=2) + '\n', encoding='utf-8')
                print('Starting FINet Vision AI with both camouflage models...')
                deadline = time.monotonic() + 60
                while not app_ready(url):
                    if process.poll() is not None:
                        print(f'FINet could not start. Read the error log: {log_path}')
                        return 1
                    if time.monotonic() >= deadline:
                        print(f'FINet is taking longer to start. Check {url} and the log: {log_path}')
                        return 1
                    time.sleep(0.25)
                print(f'FINet is ready: {url}')
                print(f'App log: {log_path}')
            if not args.no_browser and not webbrowser.open(url, new=2):
                print(f'Open this address in your browser: {url}')
            return 0
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError) as error:
        print(f'FINet launcher error: {error}')
        print('If the Python environment is unavailable, run SETUP_NEW_LAPTOP.cmd.')
        raise SystemExit(1)
