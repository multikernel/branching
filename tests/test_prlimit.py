# SPDX-License-Identifier: Apache-2.0
"""Tests for _prlimit: apply_limits via setrlimit(2)."""

import resource
from unittest.mock import patch

import pytest

from branching.process.limits import ResourceLimits
from branching.process._prlimit import apply_limits


class TestApplyLimits:
    def test_memory_sets_rlimit_as(self):
        limits = ResourceLimits(memory=512 * 1024 * 1024)
        with patch("branching.process._prlimit.resource.setrlimit") as mock:
            apply_limits(limits)
        mock.assert_called_once_with(
            resource.RLIMIT_AS,
            (512 * 1024 * 1024, 512 * 1024 * 1024),
        )

    def test_cpu_time_sets_rlimit_cpu(self):
        limits = ResourceLimits(cpu_time=60)
        with patch("branching.process._prlimit.resource.setrlimit") as mock:
            apply_limits(limits)
        mock.assert_called_once_with(resource.RLIMIT_CPU, (60, 60))

    def test_nproc_sets_rlimit_nproc(self):
        limits = ResourceLimits(nproc=100)
        with patch("branching.process._prlimit.resource.setrlimit") as mock:
            apply_limits(limits)
        mock.assert_called_once_with(resource.RLIMIT_NPROC, (100, 100))

    def test_all_fields(self):
        limits = ResourceLimits(memory=1024, cpu_time=30, nproc=10)
        with patch("branching.process._prlimit.resource.setrlimit") as mock:
            apply_limits(limits)
        assert mock.call_count == 3

    def test_none_fields_skipped(self):
        limits = ResourceLimits()
        with patch("branching.process._prlimit.resource.setrlimit") as mock:
            apply_limits(limits)
        mock.assert_not_called()

    def test_partial_fields(self):
        limits = ResourceLimits(memory=4096)
        with patch("branching.process._prlimit.resource.setrlimit") as mock:
            apply_limits(limits)
        # Only RLIMIT_AS should be set
        assert mock.call_count == 1
        assert mock.call_args[0][0] == resource.RLIMIT_AS
