# Puffer PHC

This is a simplified version of the [Perpetual Humanoid Control](https://github.com/ZhengyiLuo/PHC) repository. The class inheritance and dependencies have been greatly reduced, and [PufferLib](https://github.com/PufferAI/PufferLib) is used to train the policy. Thank you to [Zhengyi Luo](https://github.com/ZhengyiLuo) for the awesome research and open sourcing the code. 

This project was sponsored by [Puffer AI](https://puffer.ai/). If you like what you see, please contact [Puffer AI for priority service](https://puffer.ai/product.html).

This repo implements only the single primitive model, which can perform 99.0% of the AMASS 11313 motions. See the [wandb log](https://wandb.ai/kywch/pufferlib/runs/morph-93bb6039) for training details. The AMASS checkpoints are shared in the [onedrive folder](https://1drv.ms/f/c/aa549e3baa3afd7f/Esw-q2bneRZGrEAJl3tvrtABFi6m6zIHfFhkJH0Z0VOkWw?e=GWdnub).

<div float="center">
  <img src="sample_data/cmu_mocap_05_06.gif" />
</div>

## Getting Started

The default path trains on the Newton/MuJoCo backends (Python 3.12, no Isaac Gym required). Isaac Gym is still supported as a legacy backend; see [Legacy: Isaac Gym](#legacy-isaac-gym) below.

1. Clone this repository. 
    ```
    git clone https://github.com/kywch/puffer-phc.git
    ```
    Then, go to the cloned directory.
    ```
    cd puffer-phc
    ```

2. Using pixi, setup the virtual environment and install the dependencies. Install pixi, if you haven't already. See [pixi documentation](https://pixi.sh/latest/#installation) for more details. The following command is for linux.
    ```
    curl -fsSL https://pixi.sh/install.sh | bash
    ```

    The following command installs the dependencies and activates the virtual environment:
    ```
    pixi shell
    ```
    This activates the `default` environment (Python 3.12, newton + mujoco-warp, torch 2.12/cu130 — needs an NVIDIA driver that supports CUDA 13). `pixi shell` also sets `PYTHONPATH` to the repo root.

    It runs two backends: `mujoco_gpu` (mujoco-warp, batched on GPU, much faster) for training, and `mujoco_cpu` (MuJoCo C, one solver per env — not batched, so keep `--env.num-envs` small) for eval and debugging rather than as a training alternative.

3. Download the SMPL parameters from [SMPL](https://smpl.is.tue.mpg.de/), and unzip them into `smpl` folder. Rename the files `basicmodel_neutral_lbs_10_207_0_v1.1.0`, `basicmodel_m_lbs_10_207_0_v1.1.0.pkl`, `basicmodel_f_lbs_10_207_0_v1.1.0.pkl` to `SMPL_NEUTRAL.pkl`, `SMPL_MALE.pkl` and `SMPL_FEMALE.pkl`.

    For the `default` environment, also strip chumpy from the files once: see `smpl/README.md`.

4. Train a policy. `config.ini` defaults to `backend = mujoco_gpu` and `solver_preset = newton_120`:
    ```
    python scripts/train.py --config config.ini -m <MOTION FILE PATH>
    ```

    `--env.solver-preset` (mujoco backends only) selects the physics preset: `newton_120` (the default) or `newton_180` (higher fidelity, slower) for Newton-native training; `isaacgym_parity_240` (or the higher-fidelity, ~2x slower `isaacgym_parity_480`) for evaluating checkpoints trained on Isaac Gym — these two are eval-only, not meant for training. See `docs/newton_mujoco_physics.md` and `docs/physics_solver.md` for details.

    MuJoCo training presets are new; Isaac Gym is the validated training path so far (see `docs/newton_mujoco_physics.md`).

    Add `--track` to log locally to Aim under `experiments/.aim`; view with `pixi run aim up --repo experiments --host 127.0.0.1 --port 43800`. Merge runs from another machine: `pixi run aim runs --repo <other>/experiments cp --destination experiments <run_hash>`.

    To prepare your own motion data, please see the `convert_amass_data.py` script in the `scripts` folder. After conversion, you can visually inspect the data with the `vis_motion_mj.py` script.

5. Play the trained policy. This opens a live viewer and needs a display. On the mujoco backends (Newton ViewerGL) red markers on grey bodies show the reference motion; keys: R reset, T resample motions, K toggle markers, left/right arrows switch the followed env (mujoco_cpu shows only env 0):
    ```
    python scripts/train.py --mode play -c <CHECKPOINT PATH>
    ```

    Use the same solver preset the policy was trained with. For checkpoints trained on Isaac Gym, such as the shared AMASS checkpoints linked above, add `--env.solver-preset isaacgym_parity_240`.

    For batch evaluation (e.g., using 4096 envs to evaluate AMASS 11313 motions), run:
    ```
    python scripts/train.py --mode eval -c <CHECKPOINT PATH> -m <MOTION FILE PATH>
    ```

6. Sweep the hyperparameters using CARBS.
    ```
    python scripts/train.py --config config.ini --mode sweep --tag <name>
    ```

    To adjust the sweep parameters and range, edit the `config.ini` file. You might need to comment/uncomment some parts in the `train.py` script, `sweep_carbs()` function to make the sweep work.

    `--mode sweep` requires `--tag <name>`; every sweep trial is tracked and tagged in Aim. Filter in the Aim UI/search with `"<name>" in run.tags`.

7. Run the tests.
    ```
    pixi run -e default test
    ```

## Legacy: Isaac Gym

Isaac Gym (Python 3.8, pytorch 2.3.1/CUDA 12.1, Isaac Gym 1.0rc4) is kept as a legacy backend. It requires its own pixi environment and a manual Isaac Gym install.

1. Activate the `isaacgym` environment:
    ```
    pixi shell -e isaacgym
    ```

2. Install Isaac Gym and gymtorch. Download and unzip Isaac Gym from [here](https://developer.nvidia.com/isaac-gym). Then, install isaacgym by running the following command inside the virtual environment:
    ```
    cd <isaac gym directory>/python
    pip install -e .
    ```

    Also, install gymtorch by running the following command inside the repository directory:
    ```
    pixi run -e isaacgym build_gymtorch
    ```
    This gymtorch allows you to debug the isaac gym env inside VSCode.

    You can test if the installation is successful by running:
    ```
    pixi run -e isaacgym test_deps
    ```
    This should print `True`.

3. Download the SMPL parameters as in step 3 above (the original, chumpy-based `.pkl` files work as-is in the `isaacgym` environment; no re-save needed).

4. Train a policy. `config.ini` defaults to `mujoco_gpu`, so pass `--env.backend isaacgym` here and in the commands below.
    ```
    python scripts/train.py --config config.ini -m <MOTION FILE PATH> --env.backend isaacgym
    ```

5. Play the trained policy. This opens the basic Isaac Gym viewer (no reference markers) and needs a display.
    ```
    python scripts/train.py --mode play -c <CHECKPOINT PATH> --env.backend isaacgym
    ```

    For batch evaluation (e.g., using 4096 envs to evaluate AMASS 11313 motions), run:
    ```
    python scripts/train.py --mode eval -c <CHECKPOINT PATH> -m <MOTION FILE PATH> --env.backend isaacgym
    ```

6. Run the tests.
    ```
    pixi run -e isaacgym test
    ```


# Notes
* `python scripts/train.py --help` shows the full list of options for the environment and training, which allows you to override the defaults in the config file. Per-backend overrides go in `[<section>.<backend>]` (e.g. `[env.isaacgym]`, `[train.isaacgym]`); CLI flags still win.
* I tested the [style discriminator](https://arxiv.org/abs/2104.02180) in the original PHC repo and here, saw that it did not improve the imitation performance, so it is optional and off by default. To enable it, set `use_amp_obs = True` in the config, or add `--env.use-amp-obs True` to the command.
* I saw several times that "fine-tuning" pretrained weights with `-c <CHECKPOINT PATH>` resulted in much faster learning. The [L2 init reg loss](https://arxiv.org/abs/2308.11958) is being logged in the training dashboard, and you can see greater L2 distance when learning from scratch compared to fine-tuning.
* As mentioned in the [PHC paper](https://arxiv.org/abs/2305.06456), I saw that the 6-layer MLP perform better than shallow MLPs, and SiLU activations perform better than ReLUs.
* Adding LayerNorm before the last layer tamed the gradient norm, and I [swept the hyperparameters](https://wandb.ai/kywch/carbs/sweeps/fupc0sps?nw=nwuserkywch) with the max grad norm of 10 (the original PHC repo uses 50). 
* Observations are RMS normalized, but the rewards/values are not. The gamma was set to 0.98 to manage the value loss. The hyperparameter sweeps consistently converged to small lambda values, so I chose 0.2.
* Also, the sweeps consistently converged to very aggressive clip coefs (0.01) and higher learning rates. I speculate that since the trainer uses the same learning rate for both actor and critic, it's using actor clipping to slow down the actor learning relative to the critic.
* I tried using LSTM for both the actor and critic, respectively, and it didn't work better. These LSTM policies are included in the code, so feel free to try them.


# References
This repository is built on top of the following amazing repositories:

* Main PHC code, motion lib, poselib, and data scripts are from: [PHC](https://github.com/ZhengyiLuo/PHC) and [Isaac Gym](https://developer.nvidia.com/isaac-gym)
* The PPO and CARBS sweep code is from: [PufferLib](https://github.com/PufferAI/PufferLib)
* Sample motion data is from: [CMU Motion Capture Dataset](http://mocap.cs.cmu.edu/), Subject 5, Motion 6 

Please follow the license of the above repositories for usage.
