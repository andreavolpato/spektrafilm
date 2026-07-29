import pathlib
import shutil

for p in pathlib.Path(".").rglob("__pycache__"):
    shutil.rmtree(p)
