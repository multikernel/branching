# SPDX-License-Identifier: Apache-2.0
"""
BranchContext - Unified branching for speculative execution.

Supports filesystem branching (BranchFS FUSE, DaxFS), process branching
(fork + namespaces), and AI agent integration patterns (speculation,
best-of-N, reflexion, tree-of-thoughts).

Layers are loaded lazily — importing only what you need avoids pulling
in unrelated dependencies:

    from branching import Workspace          # only loads core/fs layer
    from branching import BranchContext      # only loads process layer
    from branching import Speculate          # loads agent + core layers

Example:
    from branching import Workspace

    ws = Workspace("/mnt/workspace")

    with ws.branch("attempt") as b:
        subprocess.run(["agent", "--workdir", str(b.path)])
        # auto-commits on success, auto-aborts on exception
"""

# Exceptions are lightweight and always available.
from .exceptions import (
    BranchingError,
    BranchError,
    BranchStaleError,
    BranchNotFoundError,
    CommitError,
    ConflictError,
    AbortError,
    MountError,
    ProcessBranchError,
    ForkError,
    NamespaceError,
    MemoryBranchError,
)

__all__ = [
    # Core
    "Workspace",
    "Branch",
    # Process
    "BranchContext",
    "ResourceLimits",
    # Agent patterns
    "Speculate",
    "BestOfN",
    "Reflexion",
    "TreeOfThoughts",
    "BeamSearch",
    "Tournament",
    # Results
    "SpeculationResult",
    "SpeculationOutcome",
    # Exceptions
    "BranchingError",
    "BranchError",
    "BranchStaleError",
    "BranchNotFoundError",
    "CommitError",
    "ConflictError",
    "AbortError",
    "MountError",
    "ProcessBranchError",
    "ForkError",
    "NamespaceError",
    "MemoryBranchError",
]

__version__ = "0.1.0"

# Lazy imports — each layer loads only when first accessed.
_LAZY_IMPORTS = {
    # Core (fs layer)
    "Workspace": ".core.workspace",
    "Branch": ".core.branch",
    # Process layer
    "BranchContext": ".process.context",
    "ResourceLimits": ".process.limits",
    # Agent layer
    "Speculate": ".agent.speculate",
    "BestOfN": ".agent.patterns",
    "Reflexion": ".agent.patterns",
    "TreeOfThoughts": ".agent.patterns",
    "BeamSearch": ".agent.patterns",
    "Tournament": ".agent.patterns",
    # Results
    "SpeculationResult": ".agent.result",
    "SpeculationOutcome": ".agent.result",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib
        module = importlib.import_module(_LAZY_IMPORTS[name], __name__)
        value = getattr(module, name)
        # Cache on the module so __getattr__ isn't called again.
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
