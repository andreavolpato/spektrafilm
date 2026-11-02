#!/usr/bin/env python3
"""Create a Finder-launchable app for the current Spektrafilm checkout."""

import argparse
import os
import plistlib
import shlex
import shutil
import sys
from pathlib import Path


APP_NAME = "Spektrafilm.app"
EXECUTABLE_NAME = "Spektrafilm"


def create_app(project_root: Path, destination: Path) -> Path:
    project_root = project_root.resolve(strict=True)
    destination = destination.expanduser().absolute()
    launcher = project_root / ".venv" / "bin" / "spektrafilm"
    if not launcher.is_file() or not os.access(str(launcher), os.X_OK):
        raise FileNotFoundError(
            "Could not find an executable .venv/bin/spektrafilm. "
            "Install the project dependencies in this checkout first."
        )
    icon = project_root / "src" / "spektrafilm_gui" / "assets" / "spektrafilm.icns"
    if not icon.is_file():
        raise FileNotFoundError("Could not find the Spektrafilm macOS icon at {}.".format(icon))
    if destination.exists():
        raise FileExistsError(
            "{} already exists; choose another destination or move it first.".format(destination)
        )

    contents = destination / "Contents"
    macos_dir = contents / "MacOS"
    resources_dir = contents / "Resources"
    macos_dir.mkdir(parents=True)
    resources_dir.mkdir()
    try:
        shutil.copy2(str(icon), str(resources_dir / "Spektrafilm.icns"))
        launcher_script = """#!/bin/bash
PROJECT_ROOT={project_root}

if [[ ! -d "$PROJECT_ROOT" ]]; then
    /usr/bin/osascript -e 'display dialog "无法启动 Spektrafilm：项目目录已移动。请在新目录重新创建 macOS 启动器。" buttons {{"好"}} default button 1'
    exit 1
fi

if [[ ! -x "$PROJECT_ROOT/.venv/bin/spektrafilm" ]]; then
    /usr/bin/osascript -e 'display dialog "无法启动 Spektrafilm：找不到项目虚拟环境。请在项目目录运行 uv sync，然后重新创建启动器。" buttons {{"好"}} default button 1'
    exit 1
fi

mkdir -p "$HOME/Library/Logs/Spektrafilm"
cd "$PROJECT_ROOT" || exit 1
"$PROJECT_ROOT/.venv/bin/spektrafilm" >> "$HOME/Library/Logs/Spektrafilm/launcher.log" 2>&1
status=$?
if [[ "$status" -ne 0 ]]; then
    /usr/bin/osascript -e 'display dialog "Spektrafilm 启动失败。请查看 ~/Library/Logs/Spektrafilm/launcher.log。" buttons {{"好"}} default button 1'
fi
exit "$status"
""".format(project_root=shlex.quote(str(project_root)))
        executable = macos_dir / EXECUTABLE_NAME
        executable.write_text(launcher_script, encoding="utf-8")
        executable.chmod(0o755)

        info = {
            "CFBundleDevelopmentRegion": "zh_CN",
            "CFBundleExecutable": EXECUTABLE_NAME,
            "CFBundleIconFile": "Spektrafilm",
            "CFBundleIdentifier": "org.spektrafilm.desktop",
            "CFBundleInfoDictionaryVersion": "6.0",
            "CFBundleName": "Spektrafilm",
            "CFBundlePackageType": "APPL",
            "CFBundleShortVersionString": "0.3.4",
            "CFBundleVersion": "0.3.4",
            "NSHighResolutionCapable": True,
        }
        with (contents / "Info.plist").open("wb") as plist_file:
            plistlib.dump(info, plist_file, sort_keys=True)
    except Exception:
        shutil.rmtree(destination)
        raise

    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Spektrafilm checkout containing .venv (default: this script's checkout)",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path.home() / "Applications" / APP_NAME,
        help="output app bundle (default: ~/Applications/Spektrafilm.app)",
    )
    args = parser.parse_args()

    if sys.platform != "darwin":
        parser.error("This launcher can only be created on macOS.")
    try:
        destination = create_app(args.project_root, args.destination)
    except (FileExistsError, FileNotFoundError, OSError) as exc:
        parser.error(str(exc))

    print("Created {}".format(destination))
    print("Open Spektrafilm from Finder or Launchpad.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
