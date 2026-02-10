# SPDX-License-Identifier: Apache-2.0
"""Result dataclasses for speculation outcomes."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class SpeculationResult:
    """Result from a single speculation candidate."""

    branch_index: int
    success: bool
    return_value: Any = None
    exception: Optional[Exception] = None
    exit_code: int = 0
    branch_path: Optional[Path] = None
    score: float = 0.0


@dataclass
class SpeculationOutcome:
    """Overall outcome from a speculation pattern."""

    winner: Optional[SpeculationResult] = None
    all_results: list[SpeculationResult] = field(default_factory=list)
    committed: bool = False
