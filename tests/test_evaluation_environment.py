"""Unused applications must not hide real evaluation dependency failures."""

from importlib import metadata
from types import SimpleNamespace

from scripts.check_evaluation_environment import check_required_dependencies


def resolver(packages):
    def distribution(name):
        if name not in packages:
            raise metadata.PackageNotFoundError(name)
        version, requirements = packages[name]
        return SimpleNamespace(version=version, requires=requirements)
    return distribution


def test_unrelated_package_conflicts_do_not_affect_required_graph():
    packages = {'evaluation': ('1.0', ['numpy>=2']), 'numpy': ('2.4.6', []),
                'unused_app': ('1.0', ['numpy<2'])}
    report = check_required_dependencies(['evaluation'], resolver(packages))
    assert report['errors'] == []
    assert set(report['required_versions']) == {'evaluation', 'numpy'}


def test_missing_transitive_requirement_is_rejected():
    report = check_required_dependencies(['evaluation'], resolver({'evaluation': ('1.0', ['missing>=1'])}))
    assert report['errors'] == ['Missing required dependency: missing>=1']


def test_conflicting_transitive_requirement_is_rejected():
    packages = {'evaluation': ('1.0', ['numpy<2']), 'numpy': ('2.4.6', [])}
    assert check_required_dependencies(['evaluation'], resolver(packages))['errors']


def test_requested_extra_is_checked_but_inactive_extra_is_not():
    packages = {'evaluation': ('1.0', ['extra_package; extra == "metrics"'])}
    assert not check_required_dependencies(['evaluation'], resolver(packages))['errors']
    assert check_required_dependencies(['evaluation[metrics]'], resolver(packages))['errors']


def test_cyclic_dependencies_terminate_without_omitting_constraints():
    packages = {'a': ('1.0', ['b>=1']), 'b': ('1.0', ['a>=2'])}
    report = check_required_dependencies(['a'], resolver(packages))
    assert report['errors'] == ['a>=2 requires a different version; installed 1.0']
