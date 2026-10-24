Download SMPL paramters from [SMPL](https://smpl.is.tue.mpg.de/), and unzip them into this folder. For SMPL, please download the v1.1.0 version, which contains the neutral humanoid. Rename the files `basicmodel_neutral_lbs_10_207_0_v1.1.0`, `basicmodel_m_lbs_10_207_0_v1.1.0.pkl`, `basicmodel_f_lbs_10_207_0_v1.1.0.pkl` to `SMPL_NEUTRAL.pkl`, `SMPL_MALE.pkl` and `SMPL_FEMALE.pkl`. Rename The file structure should look like this:

```
|-- smpl
    |-- SMPL_FEMALE.pkl
    |-- SMPL_NEUTRAL.pkl
    |-- SMPL_MALE.pkl
```

The default (Python 3.12) environment can't load chumpy. Convert the files' chumpy arrays to numpy once, from the repo root (chumpy needs Python <= 3.10):

```
uv run smpl/strip_chumpy.py
```

`pixi run -e isaacgym python smpl/strip_chumpy.py` also works. The converted files work in both environments.
