def preload_isaacgym(backend):
    """Import isaacgym before torch when `backend` needs it. isaacgym's gymdeps module raises if torch was
    imported first, and bypassing that check corrupts PhysX GPU state, so callers must do this before
    importing torch themselves."""
    if backend == "isaacgym":
        import isaacgym  # noqa: F401


def make_backend(name, **kwargs):
    """Import the sim backend lazily, so each one only needs its own simulator installed."""
    if name == "isaacgym":
        from puffer_phc.backend.isaacgym import IsaacGymBackend as Backend
    elif name == "mujoco_cpu":
        from puffer_phc.backend.mujoco_cpu import MujocoCPUBackend as Backend
    else:
        raise ValueError(f"Unknown backend: {name}")
    return Backend(**kwargs)
