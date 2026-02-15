# SPDX-License-Identifier: Apache-2.0
"""Implementation of 'branching status' command."""

import json

from branching import Workspace

from . import _resolve_workspace


def cmd_status(args) -> int:
    ws_path = _resolve_workspace(args)
    ws = Workspace(ws_path)

    # Enumerate branch directories (/@* convention)
    branches = []
    try:
        for entry in sorted(ws.path.iterdir()):
            if entry.name.startswith("@") and entry.is_dir():
                branches.append(entry.name[1:])  # strip leading @
    except OSError:
        pass

    data = {
        "command": "status",
        "workspace": str(ws.path),
        "fstype": ws.fstype,
        "branches": branches,
    }

    if getattr(args, "json", False):
        print(json.dumps(data))
    else:
        print(f"workspace: {ws.path}")
        print(f"fstype: {ws.fstype}")
        if branches:
            print(f"branches ({len(branches)}):")
            for name in branches:
                print(f"  {name}")
        else:
            print("branches: (none)")

    return 0
