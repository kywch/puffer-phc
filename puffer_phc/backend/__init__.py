def make_backend(name, **kwargs):
    """Import the sim backend lazily, so each one only needs its own simulator installed."""
    if name == "isaacgym":
        from puffer_phc.backend.isaacgym import IsaacGymBackend as Backend
    elif name == "mujoco_cpu":
        from puffer_phc.backend.mujoco_cpu import MujocoCPUBackend as Backend
    else:
        raise ValueError(f"Unknown backend: {name}")
    return Backend(**kwargs)
