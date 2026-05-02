try:
    from .task import ALFWorld
except ImportError:
    pass  # textworld not installed in skill-cycle environment; eval.py imports fine without it