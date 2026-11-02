from types import SimpleNamespace

import pytest

from spektrafilm_gui import app as app_module


@pytest.fixture
def launch(monkeypatch):
    events = []
    arguments = ['spektrafilm']
    controller = SimpleNamespace(
        load_input_image=lambda path: events.append(('image', path)),
        load_raw_image=lambda path: events.append(('raw', path)),
    )
    app = SimpleNamespace(viewer=object(), controller=controller, main_window=object())

    def get_qapp():
        events.append('qapp')
        return SimpleNamespace(arguments=lambda: arguments)

    def create_app():
        events.append('create')
        return app

    modules = {
        'napari': SimpleNamespace(run=lambda: events.append('run')),
        'napari.qt': SimpleNamespace(get_qapp=get_qapp),
    }
    monkeypatch.setattr(app_module, 'import_module', modules.__getitem__)
    monkeypatch.setattr(app_module, 'create_app', create_app)
    monkeypatch.setattr(app_module, 'show_viewer_window', lambda viewer: events.append('show'))
    monkeypatch.setattr(
        app_module.QtWidgets.QMessageBox,
        'critical',
        lambda parent, title, message: events.append(('error', message)),
    )
    return SimpleNamespace(arguments=arguments, events=events, controller=controller)


def test_no_input_preserves_normal_gui_launch(launch):
    app_module.main()

    assert launch.events == ['qapp', 'create', 'show', 'run']


@pytest.mark.parametrize('raw', [False, True])
def test_launch_loads_selected_image_with_spaces_and_unicode(launch, tmp_path, raw):
    image = tmp_path / ('holiday photo é.ARW' if raw else 'holiday photo é.tif')
    image.touch()
    launch.arguments.extend(['--raw', str(image)] if raw else [str(image)])

    app_module.main()

    assert launch.events == ['qapp', 'create', 'show', ('raw' if raw else 'image', str(image)), 'run']


@pytest.mark.parametrize('raw', [False, True])
def test_missing_file_is_rejected_before_creating_gui(launch, tmp_path, capsys, raw):
    path = tmp_path / 'missing.tif'
    launch.arguments.extend(['--raw', str(path)] if raw else [str(path)])

    with pytest.raises(SystemExit) as exc:
        app_module.main()

    assert exc.value.code == 2
    assert str(path) in capsys.readouterr().err
    assert 'create' not in launch.events


def test_directory_is_not_accepted_as_image(launch, tmp_path, capsys):
    launch.arguments.append(str(tmp_path))

    with pytest.raises(SystemExit) as exc:
        app_module.main()

    assert exc.value.code == 2
    assert str(tmp_path) in capsys.readouterr().err
    assert 'create' not in launch.events


@pytest.mark.parametrize('arguments', [['--unknown'], ['--raw'], ['one.tif', 'two.tif'], ['one.tif', '--raw', 'two.ARW']])
def test_invalid_arguments_are_rejected(launch, capsys, arguments):
    launch.arguments.extend(arguments)

    with pytest.raises(SystemExit) as exc:
        app_module.main()

    assert exc.value.code == 2
    assert 'usage:' in capsys.readouterr().err
    assert 'create' not in launch.events


def test_help_does_not_construct_full_gui(launch, capsys):
    launch.arguments.append('--help')

    with pytest.raises(SystemExit) as exc:
        app_module.main()

    assert exc.value.code == 0
    assert '--raw' in capsys.readouterr().out
    assert 'create' not in launch.events


def test_repeated_raw_input_is_rejected(launch, tmp_path, capsys):
    first = tmp_path / 'first.ARW'
    second = tmp_path / 'second.ARW'
    first.touch()
    second.touch()
    launch.arguments.extend(['--raw', str(first), '--raw', str(second)])

    with pytest.raises(SystemExit) as exc:
        app_module.main()

    assert exc.value.code == 2
    assert '--raw' in capsys.readouterr().err
    assert 'create' not in launch.events


@pytest.mark.parametrize('error_type', [OSError, ValueError, RuntimeError, Exception])
def test_load_failure_is_reported_and_gui_remains_usable(launch, tmp_path, error_type):
    image = tmp_path / 'broken.tif'
    image.touch()
    launch.arguments.append(str(image))

    def fail_to_load(path):
        raise error_type('unsupported image format')

    launch.controller.load_input_image = fail_to_load

    app_module.main()

    assert launch.events[-2][0] == 'error'
    assert 'unsupported image format' in launch.events[-2][1]
    assert str(image) in launch.events[-2][1]
    assert launch.events[-1] == 'run'


def test_inaccessible_file_has_actionable_argument_error(launch, tmp_path, monkeypatch, capsys):
    image = tmp_path / 'private.tif'
    launch.arguments.append(str(image))

    def deny_access(path):
        raise PermissionError('permission denied')

    monkeypatch.setattr(app_module.Path, 'is_file', deny_access)

    with pytest.raises(SystemExit) as exc:
        app_module.main()

    assert exc.value.code == 2
    assert 'permission denied' in capsys.readouterr().err
    assert 'create' not in launch.events
