try:
    import isaacgym  # noqa: F401  # must load before torch, or test_isaacgym_backend.py silently skips
except ImportError:
    pass
