import torch
import warp as wp
from pyglet.window import key

from puffer_phc.humanoid_phc import HumanoidPHC


class HumanoidRenderEnv(HumanoidPHC):
    """Play-mode env for the Newton backends. Keys: R reset, T resample motions, K toggle the reference markers
    (red), left/right arrows pick the env the camera follows."""

    def __init__(self, cfg, device_type="cuda", device_id=0, headless=False):
        cfg["env"]["state_init"] = "Start"
        cfg["env"]["motion_sampling"] = cfg["env"]["start_sampling"] = "uniform"
        super().__init__(cfg, device_type, device_id, headless)
        self.show_markers, self.view_env, self._cam_unset = True, 0, True
        model = self.backend.model  # plain grey bodies (not the ground) so the markers stand out
        color = model.shape_color.numpy()
        color[model.shape_body.numpy() >= 0] = 0.7
        model.shape_color.assign(color)
        self.backend.viewer.renderer.register_key_press(self._on_key)

    def _on_key(self, symbol, modifiers):
        n = self.backend.viewer.world_offsets.shape[0]  # worlds drawn: all envs on mujoco_gpu, env 0 on mujoco_cpu
        if symbol == key.R:
            self.reset()
        elif symbol == key.T:
            self.resample_motions()
        elif symbol == key.K:
            self.show_markers = not self.show_markers
        elif symbol in (key.LEFT, key.RIGHT):
            self.view_env = (self.view_env + (1 if symbol == key.RIGHT else -1)) % n

    def render(self):
        self.backend.render(self._draw)

    def _draw(self, viewer):
        """Called between begin_frame and end_frame: reference markers, and the camera on the viewed root."""
        offset = viewer.world_offsets.numpy()[: self.num_envs]
        offset = torch.as_tensor(offset, device=self.device)
        if self.show_markers:
            pts = (self._ref_body_pos[: len(offset)] + offset[:, None]).reshape(-1, 3).float().cpu().numpy()
            red = wp.full(
                len(pts), wp.vec3(1.0, 0.0, 0.0), dtype=wp.vec3, device=viewer.device
            )  # tuples crash the GL path
            viewer.log_points("ref_markers", wp.array(pts, dtype=wp.vec3, device=viewer.device), 0.05, red)
        else:
            viewer.log_points("ref_markers", None)
        root = (self._humanoid_root_states[self.view_env, :3] + offset[self.view_env]).tolist()
        root[2] = 0.9  # follow in x/y only: tracking the root height makes the view bob
        if self._cam_unset:  # initial view; afterwards follow() keeps the user's orbit angle and distance
            viewer.set_camera(wp.vec3(root[0], root[1] - 3.0, root[2] + 0.5), pitch=-10.0, yaw=90.0)
            viewer.camera.set_pivot(wp.vec3(*root))
            self._cam_unset = False
        viewer.camera.follow(wp.vec3(*root))
