"""Check FINet's required dependency graph without gating on unused Kaggle apps."""

import argparse
from importlib import import_module, metadata
import json
from pathlib import Path
import subprocess
import sys

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


ROOTS = ['torch>=2.6', 'torchvision>=0.21', 'numpy', 'Pillow', 'torch-dct',
         'timm', 'pytest', 'pysodmetrics==1.6.2', 'matplotlib==3.11.2', 'gdown==6.5.1']


def check_required_dependencies(requirements, distribution=metadata.distribution):
    """Check every active transitive requirement, including requested extras."""
    queue = [Requirement(value) for value in requirements]
    checked, versions, errors = set(), {}, []
    while queue:
        requirement = queue.pop()
        name = canonicalize_name(requirement.name)
        try:
            package = distribution(requirement.name)
        except metadata.PackageNotFoundError:
            errors.append(f'Missing required dependency: {requirement}')
            continue
        versions[name] = package.version
        if requirement.specifier and not requirement.specifier.contains(package.version, prereleases=True):
            errors.append(f'{requirement} requires a different version; installed {package.version}')
        identity = (name, frozenset(requirement.extras))
        if identity in checked:
            continue
        checked.add(identity)
        for value in package.requires or []:
            dependency = Requirement(value)
            extras = requirement.extras or {''}
            if dependency.marker is None or any(dependency.marker.evaluate({'extra': extra}) for extra in extras):
                queue.append(dependency)
    return {'required_versions': dict(sorted(versions.items())), 'errors': sorted(set(errors))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    report = check_required_dependencies(ROOTS)
    global_check = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True)
    report['global_pip_check'] = {'returncode': global_check.returncode,
                                'output': global_check.stdout + global_check.stderr}
    report['scope'] = ('All FINet/evaluation dependency requirements are mandatory. '
                       'The global report also includes unrelated preinstalled Kaggle applications.')
    report['import_errors'] = []
    for module in ['torch', 'torchvision', 'numpy', 'PIL', 'torch_dct', 'timm',
                   'scipy.ndimage', 'matplotlib', 'py_sod_metrics', 'gdown']:
        try:
            import_module(module)
        except Exception as error:
            report['import_errors'].append(f'{module}: {type(error).__name__}: {error}')
    report['status'] = 'passed' if not report['errors'] and not report['import_errors'] else 'failed'
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'checked_packages': len(report['required_versions']),
                      'errors': report['errors'], 'import_errors': report['import_errors'],
                      'global_pip_check_returncode': global_check.returncode, 'report': str(args.report)}))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
