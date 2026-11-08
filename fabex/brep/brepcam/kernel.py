"""Fabex 'kernel.py'

Adapter around a BRepKit kernel package. No bpy.

BRepKit can be reached in several ways (enabled Blender extension, a top-level
`brepkit` module, or a source checkout loaded by path), so the package is located
elsewhere and handed to `Kernel`. All access to kernel internals goes through this
class, so changes in BRepKit's module layout only need fixing here.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


class KernelError(Exception):
    pass


class Kernel:
    """Feature-detecting wrapper around a `brepkit` kernel package."""

    def __init__(self, package: ModuleType, source: str):
        self.package = package
        self.name = package.__name__
        # "extension" (enabled Blender extension), "module" (importable `brepkit`) or "path"
        self.source = source
        self.version = getattr(package, "__version__", "unknown")
        self.path = str(Path(package.__file__).parent) if getattr(package, "__file__", None) else ""
        self._modules: dict[str, ModuleType] = {}

    def __repr__(self) -> str:
        return f"<Kernel {self.name} {self.version} ({self.source})>"

    def module(self, sub: str) -> ModuleType:
        """Import a kernel submodule, e.g. kernel.module("io.serialize")."""
        mod = self._modules.get(sub)
        if mod is None:
            try:
                mod = importlib.import_module(f"{self.name}.{sub}")
            except ImportError as e:
                raise KernelError(f"BRepKit {self.version} has no module '{sub}': {e}") from e
            self._modules[sub] = mod
        return mod

    def has(self, sub: str, attr: str | None = None) -> bool:
        """True if the kernel provides `sub` (and `sub.attr`)."""
        try:
            mod = self.module(sub)
        except KernelError:
            return False
        return attr is None or hasattr(mod, attr)

    # --- serialization -------------------------------------------------------------------

    def loads_body(self, text: str):
        return self.module("io.serialize").loads_body(text)

    def dumps_body(self, body) -> str:
        return self.module("io.serialize").dumps_body(body)

    def info(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "source": self.source,
            "path": self.path,
        }


def _package_dir(root: str | Path) -> Path:
    """Accept either the kernel package dir (`.../brepkit`) or a checkout root containing it."""
    root = Path(root).expanduser().resolve()
    if (root / "__init__.py").is_file() and (root / "nurbs").is_dir():
        return root
    if (root / "brepkit" / "__init__.py").is_file():
        return root / "brepkit"
    raise KernelError(f"No brepkit package found at {root}")


def load_kernel_from_path(root: str | Path, alias: str = "_fabex_brepkit") -> ModuleType:
    """Load a brepkit package from disk under a private module name.

    Works because the kernel only uses relative imports. Never runs the Blender
    extension's own __init__, so it is safe when the extension is disabled.
    """
    pkg_dir = _package_dir(root)
    existing = sys.modules.get(alias)
    if existing is not None:
        if Path(getattr(existing, "__file__", "")).parent.resolve() == pkg_dir:
            return existing
        # A different checkout under the same alias: drop the old one and its submodules.
        for name in [n for n in sys.modules if n == alias or n.startswith(alias + ".")]:
            del sys.modules[name]

    spec = importlib.util.spec_from_file_location(
        alias, pkg_dir / "__init__.py", submodule_search_locations=[str(pkg_dir)]
    )
    if spec is None or spec.loader is None:
        raise KernelError(f"Cannot load brepkit from {pkg_dir}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        del sys.modules[alias]
        raise
    return module
