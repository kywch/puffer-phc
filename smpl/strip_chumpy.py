# /// script
# requires-python = "==3.10.*"
# dependencies = ["numpy==1.23.5", "chumpy"]
# [tool.uv.sources]
# chumpy = { git = "https://github.com/nico-von-huene/chumpy.git" }
# ///
"""Re-save SMPL .pkl files with chumpy arrays converted to numpy, so they load without chumpy (Python 3.12).

Needs chumpy, which fails on Python >= 3.11 (inspect.getargspec); uv reads the pinned deps from the header above:
    uv run smpl/strip_chumpy.py [PKL ...]
or in the `isaacgym` pixi env: `pixi run -e isaacgym python smpl/strip_chumpy.py [PKL ...]`.
Defaults to the SMPL_{NEUTRAL,MALE,FEMALE}.pkl files next to this script. Files without chumpy arrays are left as is.
"""

import os
import pickle
import sys

import numpy as np

SMPL_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FILES = [os.path.join(SMPL_DIR, f"SMPL_{g}.pkl") for g in ("NEUTRAL", "MALE", "FEMALE")]

for path in sys.argv[1:] or DEFAULT_FILES:
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    converted = [k for k, v in data.items() if type(v).__module__.startswith("chumpy")]
    if not converted:
        print(f"{path}: no chumpy arrays, skipped")
        continue
    for k in converted:
        data[k] = np.array(data[k])
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"{path}: converted {', '.join(converted)}")
