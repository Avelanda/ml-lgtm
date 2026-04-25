#!/usr/bin/env bash
# Clone gsplat at a pinned commit and apply the LGTM gsplat patch.
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# For third-party code see ACKNOWLEDGMENTS file.
#

#
# Clones gsplat at the pinned commit and applies the project's custom patch
# for LGTM textured 2DGS support.
#
# Usage:
#   bash scripts/clone_patch_gsplat.sh
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(dirname "$script_dir")"
cd "$project_root"

GSPLAT_REPO="https://github.com/nerfstudio-project/gsplat.git"
GSPLAT_COMMIT="32f2a54"
GSPLAT_DIR="$project_root/extern/gsplat"
GSPLAT_PATCH="$project_root/extern/gsplat.patch"

# Clone if not present.
if [ ! -d "$GSPLAT_DIR/.git" ]; then
    echo "Cloning gsplat..."
    rm -rf "$GSPLAT_DIR"
    git clone --recursive "$GSPLAT_REPO" "$GSPLAT_DIR"
fi

pushd "$GSPLAT_DIR" > /dev/null

# Reset any local modifications (patch may have already been applied).
git checkout "$GSPLAT_COMMIT"
git reset --hard "$GSPLAT_COMMIT"
git clean -fd > /dev/null
git submodule update --init --recursive

# Verify commit.
ACTUAL_COMMIT=$(git rev-parse HEAD)
if [ "${ACTUAL_COMMIT:0:7}" != "$GSPLAT_COMMIT" ]; then
    echo "Error: expected $GSPLAT_COMMIT, got $ACTUAL_COMMIT"
    exit 1
fi

# Apply patch.
if [ ! -f "$GSPLAT_PATCH" ]; then
    echo "Error: patch file not found at $GSPLAT_PATCH"
    exit 1
fi
git apply "$GSPLAT_PATCH"
echo "gsplat patched at $GSPLAT_COMMIT"

popd > /dev/null

echo "Done. To install: pip install extern/gsplat/ --no-build-isolation"
