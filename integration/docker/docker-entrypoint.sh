#!/bin/sh
set -e

# /src is the host bind-mount, /workspace is the branchfs COW layer
branchfs mount --base /src /workspace

cd /workspace
exec branching "$@"
