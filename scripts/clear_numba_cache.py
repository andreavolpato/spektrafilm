import shutil
import pathlib
for p in pathlib.Path('.').rglob('__pycache__'):
    shutil.rmtree(p)