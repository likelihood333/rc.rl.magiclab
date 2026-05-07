"""Locate the pinned backend without replacing the installed rsl_rl package."""

from pathlib import Path
import sys


def activate_backend() -> Path:
    backend_root = Path(__file__).resolve().parents[3] / "third_party" / "rsl_rl"
    if not (backend_root / "magiclab_rsl" / "__init__.py").is_file():
        raise FileNotFoundError(f"Vendored PPO backend not found: {backend_root}")
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))
    import magiclab_rsl

    loaded = Path(magiclab_rsl.__file__).resolve()
    if not loaded.is_relative_to(backend_root):
        raise RuntimeError(f"Unexpected magiclab_rsl package loaded: {loaded}")
    return backend_root
