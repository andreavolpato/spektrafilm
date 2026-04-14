from importlib import import_module

__all__ = [
    'build_qt_snapshot_viewer',
    'clear_diagnostic_profile_snapshots',
    'create_qt_diagnostic_profile_snapshot_viewer',
    'get_diagnostic_profile_snapshots',
    'launch_qt_snapshot_viewer',
    'list_diagnostic_profile_snapshots',
    'log_event',
    'log_parameters',
    'plot_diagnostic_profile_snapshot',
]


def __getattr__(name: str):
    if name in {'clear_diagnostic_profile_snapshots', 'get_diagnostic_profile_snapshots', 'log_event', 'log_parameters'}:
        module = import_module('spektrafilm_profile_creator.diagnostics.messages')
        return getattr(module, name)
    if name in {
        'build_qt_snapshot_viewer',
        'create_qt_diagnostic_profile_snapshot_viewer',
        'launch_qt_snapshot_viewer',
        'list_diagnostic_profile_snapshots',
        'plot_diagnostic_profile_snapshot',
    }:
        module = import_module('spektrafilm_profile_creator.diagnostics.snapshot_viewer')
        return getattr(module, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')