import os
import plistlib
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.create_macos_app import create_app


class CreateMacOSAppTests(unittest.TestCase):
    def test_creates_app_bundle_for_project_environment(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "project with spaces"
            root.mkdir()
            launcher = root / ".venv" / "bin" / "spektrafilm"
            launcher.parent.mkdir(parents=True)
            launcher.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            launcher.chmod(0o755)
            destination = root / "Applications" / "Spektrafilm.app"

            app = create_app(root, destination)

            executable = app / "Contents" / "MacOS" / "Spektrafilm"
            self.assertTrue(executable.stat().st_mode & 0o111)
            self.assertIn(str(root), executable.read_text(encoding="utf-8"))
            with (app / "Contents" / "Info.plist").open("rb") as plist_file:
                info = plistlib.load(plist_file)
            self.assertEqual(info["CFBundlePackageType"], "APPL")
            self.assertEqual(info["CFBundleExecutable"], "Spektrafilm")

    def test_app_launcher_runs_the_project_entry_point(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "project with spaces"
            root.mkdir()
            home = Path(temporary_directory) / "home"
            marker = Path(temporary_directory) / "launched"
            launcher = root / ".venv" / "bin" / "spektrafilm"
            launcher.parent.mkdir(parents=True)
            launcher.write_text(
                "#!/bin/sh\nprintf started > \"$MARKER_PATH\"\n",
                encoding="utf-8",
            )
            launcher.chmod(0o755)
            app = create_app(root, root / "Applications" / "Spektrafilm.app")
            environment = os.environ.copy()
            environment["HOME"] = str(home)
            environment["MARKER_PATH"] = str(marker)

            subprocess.run(
                ["bash", str(app / "Contents" / "MacOS" / "Spektrafilm")],
                check=True,
                env=environment,
            )

            self.assertEqual(marker.read_text(encoding="utf-8"), "started")

    def test_requires_an_installed_project_launcher(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with self.assertRaises(FileNotFoundError):
                create_app(root, root / "Spektrafilm.app")


if __name__ == "__main__":
    unittest.main()
