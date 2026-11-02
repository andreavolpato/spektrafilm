from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import tempfile
import tomllib
import zipfile

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_VERSION = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))['project']['version']
SOURCE_FILES = (
    'CITATION.cff',
    'LICENSE',
    'README.md',
    'SPEKTRAFILM_LICENSE.txt',
    'pyproject.toml',
)


def build_package(output: Path) -> Path:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix='spektrafilm-windows-') as temporary:
        staging = Path(temporary)
        shutil.copytree(
            ROOT / 'src',
            staging / 'Source' / 'src',
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'),
        )
        for filename in SOURCE_FILES:
            shutil.copy2(ROOT / filename, staging / 'Source' / filename)

        scripts = ROOT / 'scripts' / 'windows'
        for filename in (
            'Install-Spektrafilm.bat',
            'Install-Spektrafilm.ps1',
            'Launch-Spektrafilm.bat',
        ):
            shutil.copy2(scripts / filename, staging / filename)

        logo = ROOT / 'src' / 'spektrafilm_gui' / 'assets' / 'spektrafilm_app_icon.png'
        with Image.open(logo) as image:
            image.convert('RGBA').save(
                staging / 'Spektrafilm.ico',
                format='ICO',
                sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
            )

        with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(staging.rglob('*')):
                if path.is_file():
                    archive.write(path, path.relative_to(staging).as_posix())

    return output


def main() -> None:
    parser = argparse.ArgumentParser(description='Build the Windows setup ZIP for Spektrafilm.')
    parser.add_argument(
        '--output',
        type=Path,
        default=ROOT / 'dist' / f'Spektrafilm-Windows-{PACKAGE_VERSION}.zip',
    )
    arguments = parser.parse_args()
    print(build_package(arguments.output))


if __name__ == '__main__':
    main()
